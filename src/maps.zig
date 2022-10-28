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

// Configuration:
pub const sample_frequency = 44_100;
pub const panSample = pan_laws.constantPower;
pub const Sample = f32;

// Implemenation:

pub const Channel = enum {
    left,
    right,
};

pub const Mixer = struct {
    pub const max_streams = 16;

    streams: std.BoundedArray(Stream, max_streams) = .{},
    next_handle: u32 = 0,
    lock: std.Thread.Mutex = .{},

    time: u64 = 0,

    pub const RepeatCount = union(enum) { once, repeat: u32, forever };
    pub fn play(mixer: *Mixer, source: AudioSource, start_offset: ?u64, count: RepeatCount) error{NoStreamLeft}!StreamHandle {
        std.debug.assert(start_offset == null); // TODO: not implemented yet!
        const stream = mixer.streams.addOne() catch return error.NoStreamLeft;
        stream.* = Stream{
            .handle = @intToEnum(StreamHandle, mixer.next_handle),
            .source = source,
            .offset = 0,
            .repetition = switch (count) {
                .once => 1,
                .repeat => |t| t,
                .forever => null,
            },
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

    /// Converts a time stamp (in seconds) to an offset in samples.
    pub fn timeToOffset(mixer: Mixer, time_in_s: f64) u64 {
        _ = mixer;
        const samples = sample_frequency * time_in_s;
        return @floatToInt(u64, samples);
    }

    /// Plays back the currently active streams and renders the output into `left_buffer` and `right_buffer`.
    /// Also advances time by the number of samples in the buffers.
    /// Both buffers must have the same number of samples in them.
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
                stream_index += 1;
                continue;
            }

            // TODO: Implement stream.pitch handling

            const volume = panSample(.{
                .left = stream.volume,
                .right = stream.volume,
            }, stream.pan);

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
                    left_buffer[sample_offset + i] += sample * volume.left;
                }
                for (right_scratch[0..count]) |sample, i| {
                    right_buffer[sample_offset + i] += sample * volume.right;
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
        mixer.time += left_buffer.len;
    }
};

pub const pan_laws = struct {
    //! For more information, see
    //! http://www.cs.cmu.edu/~music/icm-online/readings/panlaws/

    /// Pans a audio sample between left and right, returns the panned sample
    /// `pan` is between -1 and 1
    pub const Pan = fn (LR_Sample, pan: f32) LR_Sample;

    /// trivial implementation, not good audio qualities.
    pub fn linear(sample: LR_Sample, pan: f32) LR_Sample {
        return LR_Sample{
            .left = sample.left * (0.5 - 0.5 * pan),
            .right = sample.right * (0.5 + 0.5 * pan),
        };
    }

    /// Use when speakers are not parallel
    pub fn constantPower(sample: LR_Sample, pan: f32) LR_Sample {
        const theta = (std.math.pi / 2.0) * (0.5 + 0.5 * pan);
        return LR_Sample{
            .left = sample.left * @cos(theta),
            .right = sample.right * @sin(theta),
        };
    }

    /// Use when speakers are parallel aligned
    pub fn @"-4.5dB"(sample: LR_Sample, pan: f32) LR_Sample {
        const l = 0.5 + 0.5 * pan;
        const theta = l * std.math.pi / 2.0;

        return LR_Sample{
            .left = sample.left * @sqrt((1.0 - l) * @cos(theta)),
            .right = sample.right * @sqrt(l * @sin(theta)),
        };
    }
};

pub const LR_Sample = struct {
    left: Sample,
    right: Sample,
};

pub const StreamHandle = enum(u32) {
    _,
};

pub const Stream = struct {
    handle: StreamHandle,

    source: AudioSource,
    offset: usize = 0,

    paused: bool = false,
    repetition: ?u32,

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

        const waves_per_sample = std.math.tau * sine.frequency / sample_frequency;

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

    const handle = try mixer.play(file.source(), null, .once);

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

test "Mixer.mix" {
    var sine_440 = SineSource{ .frequency = 440 };
    var sine_1000 = SineSource{ .frequency = 1000 };

    var sound_file = try SoundFile.initRawMono(std.testing.allocator, &.{ 0, 0, 0, 1, 0, 0, 0, -1 });
    defer sound_file.deinit();

    var mixer = Mixer{};

    _ = try mixer.play(sine_440.source(), null, .once);
    _ = try mixer.play(sine_1000.source(), null, .once);
    _ = try mixer.play(sound_file.source(), null, .once);

    var left: [4096]f32 = undefined;
    var right: [4096]f32 = undefined;

    const cwd = std.fs.cwd();
    var file = try cwd.createFile("sine.wav", .{});
    defer file.close();

    const num_chunks: usize = 32;

    var file_source = std.io.StreamSource{ .file = file };

    var wave = try WaveFileWriter.begin(&file_source);

    var i: usize = 0;
    while (i < num_chunks) : (i += 1) {
        mixer.mix(&left, &right);

        try wave.write(&left, &right);
    }

    try wave.end();
}

test "Mixer.play, .once" {
    var sound_file = try SoundFile.initRawMono(std.testing.allocator, &.{ 1, 0, 0, 0 });
    defer sound_file.deinit();

    var mixer = Mixer{};

    _ = try mixer.play(sound_file.source(), null, .once);

    var left: [64]f32 = undefined;
    var right: [64]f32 = undefined;

    mixer.mix(&left, &right);

    var lc: usize = 0;
    for (left) |s| {
        if (s > 0.1) lc += 1;
    }
    var rc: usize = 0;
    for (right) |s| {
        if (s > 0.1) rc += 1;
    }

    try std.testing.expectEqual(@as(usize, 1), lc);
    try std.testing.expectEqual(@as(usize, 1), rc);
}

test "Mixer.play, .repeat" {
    var sound_file = try SoundFile.initRawMono(std.testing.allocator, &.{ 1, 0, 0, 0 });
    defer sound_file.deinit();

    var mixer = Mixer{};

    _ = try mixer.play(sound_file.source(), null, .{ .repeat = 3 });

    var left: [64]f32 = undefined;
    var right: [64]f32 = undefined;

    mixer.mix(&left, &right);

    var lc: usize = 0;
    for (left) |s| {
        if (s > 0.1) lc += 1;
    }
    var rc: usize = 0;
    for (right) |s| {
        if (s > 0.1) rc += 1;
    }

    try std.testing.expectEqual(@as(usize, 3), lc);
    try std.testing.expectEqual(@as(usize, 3), rc);
}

test "Mixer.play, .forever" {
    var sound_file = try SoundFile.initRawMono(std.testing.allocator, &.{ 1, 0, 0, 0 });
    defer sound_file.deinit();

    var mixer = Mixer{};

    _ = try mixer.play(sound_file.source(), null, .forever);

    var left: [64]f32 = undefined;
    var right: [64]f32 = undefined;

    mixer.mix(&left, &right);

    var lc: usize = 0;
    for (left) |s| {
        if (s > 0.1) lc += 1;
    }
    var rc: usize = 0;
    for (right) |s| {
        if (s > 0.1) rc += 1;
    }

    try std.testing.expectEqual(@as(usize, left.len / 4), lc);
    try std.testing.expectEqual(@as(usize, right.len / 4), rc);
}

const WaveFileWriter = struct {
    pub const Format = enum(u16) {
        pcm = 0x0001, // 	PCM
        ieee_float = 0x0003, // 	IEEE float
        alaw = 0x0006, // 	8-bit ITU-T G.711 A-law
        mulaw = 0x0007, // 	8-bit ITU-T G.711 µ-law
        extensible = 0xFFFE, //
    };

    const PCM_Sample = i16;

    stream: *std.io.StreamSource,

    riff_chunk: Chunk,
    data_chunk: Chunk,

    pub fn begin(stream: *std.io.StreamSource) !WaveFileWriter {
        const channels = 2;

        try stream.seekTo(0); // make sure we're at the front

        var writer = stream.writer();

        // Start chunk:
        const riff_chunk = try beginChunk(stream, "RIFF");

        try writer.writeAll("WAVE");

        const fmt_chunk = try beginChunk(stream, "fmt ");

        try writer.writeIntLittle(u16, @enumToInt(Format.pcm)); // wFormatTag
        try writer.writeIntLittle(u16, channels); // nChannels
        try writer.writeIntLittle(u32, sample_frequency); // nSamplesPerSec

        try writer.writeIntLittle(u32, channels * sample_frequency * @sizeOf(PCM_Sample)); // nAvgBytesPerSec	Data rate
        try writer.writeIntLittle(u16, channels * @sizeOf(PCM_Sample)); // nBlockAlign, //  Data block size (bytes)
        try writer.writeIntLittle(u16, @bitSizeOf(PCM_Sample)); // wBitsPerSample, //  Bits per sample

        try endChunk(stream, fmt_chunk);

        const data_chunk = try beginChunk(stream, "data");

        return WaveFileWriter{
            .stream = stream,
            .riff_chunk = riff_chunk,
            .data_chunk = data_chunk,
        };
    }

    fn mapToSample(in: f32) PCM_Sample {
        const lo = std.math.minInt(PCM_Sample);
        const hi = std.math.maxInt(PCM_Sample);

        return @floatToInt(PCM_Sample, std.math.clamp(
            (hi - lo) * (0.5 + 0.5 * in) + lo,
            lo,
            hi,
        ));
    }

    pub fn write(wave: *WaveFileWriter, left: []const Sample, right: []const Sample) !void {
        var buffered_writer = std.io.bufferedWriter(wave.stream.writer());

        std.debug.assert(left.len == right.len);
        for (left) |l, i| {
            const r = right[i];
            try buffered_writer.writer().writeIntLittle(PCM_Sample, mapToSample(l));
            try buffered_writer.writer().writeIntLittle(PCM_Sample, mapToSample(r));
        }

        try buffered_writer.flush();
    }

    pub fn end(self: *WaveFileWriter) !void {
        try endChunk(self.stream, self.data_chunk);
        try endChunk(self.stream, self.riff_chunk);
    }

    const Chunk = enum(u64) { _ };

    fn beginChunk(out: *std.io.StreamSource, id: *const [4]u8) !Chunk {
        try out.writer().writeAll(id);
        const pos = @intToEnum(Chunk, try out.getPos());
        try out.writer().writeIntLittle(u32, 0xAAAAAAAA); // write bogus data
        return pos;
    }

    fn endChunk(out: *std.io.StreamSource, chunk: Chunk) !void {
        const chunk_start = @enumToInt(chunk);
        const chunk_end = try out.getPos();
        try out.seekTo(chunk_start);
        try out.writer().writeIntLittle(u32, std.math.cast(u32, chunk_end - chunk_start - 4) orelse return error.Overflow);
        try out.seekTo(chunk_end);
    }
};
