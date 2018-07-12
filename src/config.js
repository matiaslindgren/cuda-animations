"use strict";

function assert(expr, msg) {
    if (!expr) {
        console.error("ASSERTION FAILED", msg);
        pause();
    }
}

function printobj(o) {
    console.log(JSON.stringify(o));
}

const CONFIG = {
    animation: {
        // Delay in ms between rendering each frame
        drawDelayMS: 1000.0 / 420.0,
        SMCycleLabelSize: 20,
        kernelSourceTextSize: 16,
        kernelSourceTextHeight: 22,
        kernelHighlightPalette: [
            [[150, 50, 50, 0.15],
             [200, 50, 50, 0.15],
             [250, 50, 50, 0.15]],
            [[50, 150, 50, 0.15],
             [50, 200, 50, 0.15],
             [50, 250, 50, 0.15]],
            [[50, 50, 150, 0.15],
             [50, 50, 200, 0.15],
             [50, 50, 250, 0.15]],
        ],
    },
    latencies: {
        arithmetic: 10,
        L2CacheAccess: 25,
        memoryAccess: 100,
    },
    memory: {
        // Amount of indexable memory slots on each row and column
        rowSlotCount: 32,
        columnSlotCount: 32,
        // Empty space between each slot on all sides
        slotPadding: 1,
        slotSize: 10,
        slotFillRGBA: [100, 100, 100, 0.15],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 15,
    },
    cache: {
        // Size of a L2 cacheline in words
        L2CacheLineSize: 8,
        L2CacheLines: 16,
        cachedStateRGBA: [60, 60, 240, 0.5],
        pendingStateRGBA: [60, 60, 240, 0.2],
    },
    SM: {
        count: 3,
        warpSize: 32,
        warpSchedulers: 2,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 1,
        paddingX: 20,
        paddingY: 20,
        height: 105,
        fillRGBA: [100, 100, 100, 0.1],
    },
    grid: {
        dimGrid: {x: Math.round(32*32/64)},
        dimBlock: {x: 64},
    },
};
