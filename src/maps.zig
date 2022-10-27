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

pub const sample_frequency = 44_100;

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

        std.debug.print("mix {} samples, {} streams\n", .{ left_buffer.len, mixer.streams.len });

        var left_scratch_buffer: [256]Sample = undefined;
        var right_scratch_buffer: [256]Sample = undefined;

        var stream_index: usize = 0;
        next_stream: while (stream_index < mixer.streams.len) {
            const stream = &mixer.streams.buffer[stream_index];
            if (stream.paused) {
                std.debug.print("render stream {} ({})\n", .{ stream_index, stream.handle });
                stream_index += 1;
                continue;
            }
            std.debug.print("render stream {} ({})\n", .{ stream_index, stream.handle });

            // TODO: Implement stream.pitch handling

            // basic linear panning
            const left_vol = stream.volume * (0.5 - 0.5 * stream.pan);
            const right_vol = stream.volume * (0.5 + 0.5 * stream.pan);

            var sample_offset: usize = 0;
            while (sample_offset < left_buffer.len) {
                const increment = std.math.min(left_buffer.len - sample_offset, left_scratch_buffer.len);
                defer sample_offset += increment;

                const left_scratch = left_scratch_buffer[0..increment];
                const right_scratch = right_scratch_buffer[0..increment];

                const count = stream.source.fetch(stream.offset, left_scratch, right_scratch);
                if (count == 0) {
                    // quick opt-out prevents us from mixing silent data
                    std.debug.print("kill stream {} ({})\n", .{ stream_index, stream.handle });
                    _ = mixer.streams.swapRemove(stream_index);
                    continue :next_stream;
                }

                for (left_scratch[0..count]) |sample, i| {
                    left_buffer[sample_offset + i] += sample * left_vol;
                }
                for (right_scratch[0..count]) |sample, i| {
                    right_buffer[sample_offset + i] += sample * right_vol;
                }

                if (count < left_scratch.len) {
                    std.debug.print("kill stream {} ({})\n", .{ stream_index, stream.handle });
                    _ = mixer.streams.swapRemove(stream_index);
                    continue :next_stream;
                }

                stream.offset += count;
            }

            // we still had samples left
            stream_index += 1;
        }
        std.debug.print("done mixing {} streams\n", .{stream_index});
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
        return @ptrCast(*T, @alignCast(@alignOf(T), source.erased));
    }

    pub fn fetch(source: AudioSource, offset_hint: usize, left_samples: []Sample, right_samples: []Sample) usize {
        std.debug.assert(left_samples.len == right_samples.len);
        return source.vtable.fetchPtr(source, offset_hint, left_samples, right_samples);
    }
};

pub const SineSource = struct {
    const vtable = AudioSource.VTable{
        .fetchPtr = fetchSamples,
    };

    frequency: f32 = 440,
    pub fn source(file: *SineSource) AudioSource {
        return AudioSource.init(file, &vtable);
    }

    fn fetchSamples(src: AudioSource, offset_hint: usize, left_samples: []Sample, right_samples: []Sample) usize {
        const sine = src.cast(SineSource);

        const waves_per_sample = 2.0 * std.math.tau * sine.frequency / sample_frequency;

        var time = waves_per_sample * @intToFloat(f32, offset_hint);

        for (left_samples[0..]) |*left, i| {
            const right = &right_samples[i];

            const sample = @sin(time);
            left.* = sample;
            right.* = sample;

            time += waves_per_sample;
        }

        return left_samples.len;
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

test "mixer basic mix" {
    var sine_440 = SineSource{ .frequency = 440 };
    var sine_1000 = SineSource{ .frequency = 1000 };

    var mixer = Mixer{};

    _ = try mixer.play(sine_440.source());
    _ = try mixer.play(sine_1000.source());

    var left: [4096]f32 = undefined;
    var right: [4096]f32 = undefined;

    const cwd = std.fs.cwd();
    var file = try cwd.createFile("sine.wav", .{});
    defer file.close();

    const num_chunks: usize = 32;
    try WaveFileWriter.writeHeaders(file.writer(), num_chunks * left.len);

    var i: usize = 0;
    while (i < num_chunks) : (i += 1) {
        mixer.mix(&left, &right);

        try WaveFileWriter.writeAudio(file.writer(), &left, &right);
    }
}

const WaveFileWriter = struct {
    const File = std.fs.File;

    const channels = 1;
    const HEADER_SIZE = 36;
    const SUBCHUNK1_SIZE = 16;
    const AUDIO_FORMAT = 3; // f32
    const BIT_DEPTH = 32; // bit
    const BYTE_SIZE = 8;
    const PI = 3.14159265358979323846264338327950288;

    fn writeHeaders(file: File.Writer, num_samples: u32) !void {
        try file.writeAll("RIFF");
        try file.writeIntLittle(u32, HEADER_SIZE + num_samples);
        try file.writeAll("WAVEfmt ");
        try file.writeIntLittle(u32, SUBCHUNK1_SIZE);
        try file.writeIntLittle(u16, AUDIO_FORMAT);
        try file.writeIntLittle(u16, channels);
        try file.writeIntLittle(u32, sample_frequency);
        try file.writeIntLittle(u32, sample_frequency * channels * (BIT_DEPTH / BYTE_SIZE));
        try file.writeIntLittle(u16, (channels * (BIT_DEPTH / BYTE_SIZE)));
        try file.writeIntLittle(u16, BIT_DEPTH);
        try file.writeAll("data");
        try file.writeIntLittle(u32, num_samples * channels * (BIT_DEPTH / BYTE_SIZE));
    }

    fn writeAudio(file: File.Writer, left: []const Sample, right: []const Sample) !void {
        std.debug.assert(left.len == right.len);
        for (left) |l, i| {
            const r = right[i];
            try file.writeIntLittle(u32, @bitCast(u32, l));
            try file.writeIntLittle(u32, @bitCast(u32, r));
        }
    }
};
