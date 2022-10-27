//! Library concepts:
//!
//! Audio data comes from a AudioSource, and is mixed in a Mixer
//! via Streams. Each stream plays back a single AudioSource
//! with offset, pitch, pan and volume.
//!
//! The stream fetches audio samples from its source as necessary,
//! processing the samples and mixing them into a global context.
//!

const std = @import("std");

pub const Sample = f32;

pub const Channel = enum {
    left,
    right,
};

pub const Mixer = struct {
    pub const max_streams = 16;

    streams: std.BoundedArray(Stream, max_streams) = .{},
    next_handle: u32 = 0,
    lock: std.Thread.Mutex = .{},

    pub fn play(mixer: *Mixer, source: AudioSource) error{NoStreamLeft}!StreamHandle {
        const stream = mixer.streams.addOne() catch return error.NoStreamLeft;
        stream.* = Stream{
            .handle = @intToEnum(StreamHandle, mixer.next_handle),
            .source = source,
            .offset = 0,
        };
        mixer.next_handle += 1; // assume we'll never have an overflow here
        return stream.handle;
    }

    pub fn pause(mixer: *Mixer, handle: StreamHandle) error{NotFound}!void {
        // interlocked access to the stream array
        mixer.lock.lock();
        defer mixer.lock.unlock();

        const stream = try mixer.resolve(handle);
        stream.paused = true;
    }

    pub fn start(mixer: *Mixer, handle: StreamHandle) error{NotFound}!void {
        // interlocked access to the stream array
        mixer.lock.lock();
        defer mixer.lock.unlock();

        const stream = try mixer.resolve(handle);
        stream.paused = false;
    }

    pub fn stop(mixer: *Mixer, handle: StreamHandle) error{NotFound}!void {
        // interlocked access to the stream array
        mixer.lock.lock();
        defer mixer.lock.unlock();

        const stream = try mixer.resolve(handle);
        const index = (@ptrToInt(stream) - @ptrToInt(&mixer.streams.buffer)) / @sizeOf(Stream);
        _ = mixer.streams.swapRemove(index);
    }

    pub fn isPaused(mixer: *Mixer, handle: StreamHandle) error{NotFound}!bool {
        // interlocked access to the stream array
        mixer.lock.lock();
        defer mixer.lock.unlock();

        const stream = try mixer.resolve(handle);
        return stream.paused;
    }

    fn resolve(mixer: *Mixer, handle: StreamHandle) error{NotFound}!*Stream {
        return for (mixer.streams.slice()) |*stream| {
            if (stream.handle == handle)
                break stream;
        } else error.NotFound;
    }

    pub fn mix(mixer: *Mixer, left_buffer: []Sample, right_buffer: []Sample) void {
        std.debug.assert(left_buffer.len == right_buffer.len);

        std.mem.set(Sample, left_buffer, 0.0);
        std.mem.set(Sample, right_buffer, 0.0);

        var left_scratch_buffer: [256]Sample = undefined;
        var right_scratch_buffer: [256]Sample = undefined;

        var stream_index: usize = 0;
        while (stream_index < mixer.streams.len) {
            const stream = &mixer.streams.buffer[stream_index];
            if (stream.paused) {
                stream_index += 1;
                continue;
            }

            // TODO: Implement stream.pitch handling

            // basic linear panning
            const left_vol = stream.volume * (0.5 - 0.5 * stream.pan);
            const right_vol = stream.volume * (0.5 + 0.5 * stream.pan);

            var sample_offset: usize = 0;
            while (sample_offset < left_buffer.len) {
                const increment = std.math.min(left_buffer.len - sample_offset, left_scratch_buffer.len);

                const left_scratch = left_scratch_buffer[0..increment];
                const right_scratch = right_scratch_buffer[0..increment];

                const count = stream.source.fetch(stream.offset, left_scratch, right_scratch);
                if (count == 0) {
                    // quick opt-out prevents us from mixing silent data
                    mixer.streams.swapRemove(stream_index);
                    continue;
                }

                for (left_scratch[0..count]) |sample, i| {
                    left_buffer[i] += sample * left_vol;
                }
                for (right_scratch[0..count]) |sample, i| {
                    right_buffer[i] += sample * right_vol;
                }

                if (count < left_scratch.len) {
                    mixer.streams.swapRemove(stream_index);
                } else {
                    // we still have samples left
                    stream_index += 1;
                }
            }
        }
    }
};

pub const StreamHandle = enum(u32) {
    _,
};

pub const Stream = struct {
    handle: StreamHandle,

    source: AudioSource,
    offset: usize = 0,

    paused: bool = false,

    volume: f32 = 1.0, // [0...1]
    pitch: f32 = 1.0, // [0...*]
    pan: f32 = 0.0, // [-1...1] (negative=left, positive=right)
};

pub const AudioSource = struct {
    pub const VTable = struct {
        fetchPtr: *const fn (AudioSource, usize, []Sample, []Sample) usize,
    };

    erased: *anyopaque,
    vtable: *const VTable,

    pub fn init(any: anytype, vtable: *const VTable) AudioSource {
        return AudioSource{
            .erased = @ptrCast(*anyopaque, any),
            .vtable = vtable,
        };
    }

    pub fn cast(source: AudioSource, comptime T: type) *T {
        return @ptrCast(*SoundFile, @alignCast(@alignOf(SoundFile), source.erased));
    }

    pub fn fetch(source: AudioSource, offset_hint: usize, left_samples: []Sample, right_samples: []Sample) usize {
        std.debug.assert(left_samples.len == right_samples.len);
        return source.vtable.fetchPtr(source.erased, offset_hint, left_samples, right_samples);
    }
};

pub const SoundFile = struct {
    const vtable = AudioSource.VTable{
        .fetchPtr = fetchSamples,
    };

    allocator: std.mem.Allocator,
    mono: bool = false,
    samples: []Sample, // stream of either linear mono samples or interleaved stereo samples (L,R,L,R,L,R,…)

    pub fn initRawMono(allocator: std.mem.Allocator, samples: []const Sample) !SoundFile {
        const clone = try allocator.dupe(Sample, samples);
        return SoundFile{
            .allocator = allocator,
            .mono = true,
            .samples = clone,
        };
    }

    pub fn initRawStereo(allocator: std.mem.Allocator, samples: []const Sample) !SoundFile {
        std.debug.assert((samples.len % 2) == 0);
        const clone = try allocator.dupe(Sample, samples);
        return SoundFile{
            .allocator = allocator,
            .mono = false,
            .samples = clone,
        };
    }

    pub fn deinit(file: *SoundFile) void {
        file.allocator.free(file.samples);
        file.* = undefined;
    }

    pub fn source(file: *SoundFile) AudioSource {
        return AudioSource.init(file, &vtable);
    }

    fn fetchSamples(src: AudioSource, offset_hint: usize, left_samples: []Sample, right_samples: []Sample) usize {
        const file = src.cast(SoundFile);

        switch (file.mono) {
            inline else => |mono| {
                const items = if (mono) 1 else 2;

                const available_rest = (file.samples.len / items) - offset_hint;
                const fetched_samples = std.math.min(left_samples.len, available_rest);
                var offset: usize = items * offset_hint;

                for (left_samples[0..fetched_samples]) |*left, i| {
                    const right = &right_samples[i];

                    if (mono) {
                        const sample = file.samples[offset];
                        left.* = sample;
                        right.* = sample;
                    } else {
                        left.* = file.samples[offset + 0];
                        right.* = file.samples[offset + 1];
                    }

                    offset += items;
                }

                return fetched_samples;
            },
        }
    }
};

// snd_create(STRING* filename): SOUND*
// snd_createoal(STRING* filename): SOUND*
// snd_createstream(STRING* filename): SOUND*

// ent_playsound ( ENTITY*, SOUND*, var volume): handle
// ent_playsound2 ( ENTITY*, SOUND*, var volume, var range): handle
// ent_playloop ( ENTITY*, SOUND*, var volume): handle
// ent_playloop2 ( ENTITY*, SOUND*, var volume, var range): handle

// snd_play ( SOUND*, var Volume, var Balance): handle
// snd_loop ( SOUND*, var Volume, var Balance): handle

// snd_stopall(mode);

// snd_tune ( var Handle, var Volume, var Freqency, var Balance);
// snd_cone (var handle, ANGLE* angle, var* cone)

// snd_add(var handle, var offset, void* Sample, var length)
// snd_buffer(SOUND* snd, void** pDesc, void*** ppSample)

test "mixer instantiation" {
    var mixer = Mixer{};
    _ = mixer;
}

test "sound file mono instantiation" {
    var file = try SoundFile.initRawMono(std.testing.allocator, &.{ 0.0, 1.0, 0.0, -1.0 });
    defer file.deinit();
}

test "sound file stereo instantiation" {
    var file = try SoundFile.initRawStereo(std.testing.allocator, &.{
        0.0,  0.0,
        1.0,  -1.0,
        0.0,  0.0,
        -1.0, 1.0,
    });
    defer file.deinit();
}

test "mixer basic stream setup" {
    var file = try SoundFile.initRawMono(std.testing.allocator, &.{ 0.0, 1.0, 0.0, -1.0 });
    defer file.deinit();

    var mixer = Mixer{};

    const handle = try mixer.play(file.source());

    try std.testing.expectEqual(false, try mixer.isPaused(handle));
    try mixer.pause(handle);
    try std.testing.expectEqual(true, try mixer.isPaused(handle));
    try mixer.start(handle);
    try std.testing.expectEqual(false, try mixer.isPaused(handle));
    try mixer.stop(handle);

    try std.testing.expectError(error.NotFound, mixer.pause(handle));
    try std.testing.expectError(error.NotFound, mixer.start(handle));
    try std.testing.expectError(error.NotFound, mixer.stop(handle));
    try std.testing.expectError(error.NotFound, mixer.isPaused(handle));
}
