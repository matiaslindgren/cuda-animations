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
        drawDelayMS: 1000.0 / 30.0,
        SMCycleLabelSize: 20,
        kernelSourceTextSize: 16,
        kernelSourceTextHeight: 22,
    },
    latencies: {
        arithmetic: 10,
        L2CacheAccess: 30,
        memoryAccess: 60,
    },
    memory: {
        // Amount of indexable memory slots on each row and column
        rowSlotCount: 32,
        columnSlotCount: 32,
        // Empty space between each slot on all sides
        slotPadding: 2,
        slotSize: 23,
        slotFillRGBA: [100, 100, 100, 0.15],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 10,
    },
    cache: {
        // Size of a L2 cacheline in words
        L2CacheLineSize: 8,
        L2CacheLines: 16,
        cachedStateRGBA: [100, 100, 220, 0.5],
    },
    SM: {
        count: 2,
        warpSize: 32,
        warpSchedulers: 2,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 3,
        paddingX: 20,
        paddingY: 20,
        height: 155,
        fillRGBA: [100, 100, 100, 0.1],
    },
    grid: {
        dimGrid: {x: Math.round(32*32/64)},
        dimBlock: {x: 64},
    },
};
