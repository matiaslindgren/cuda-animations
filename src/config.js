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

function generatePaletteRGBA(shadeCount) {
    const baseRGBA = [50, 50, 50, 0.15];
    const shadeIncrement = (255 - 50) / shadeCount;
    return Array.from(new Array(3), (_, component) => {
        return Array.from(new Array(shadeCount), (_, shade) => {
            const color = baseRGBA.slice();
            color[component] = 255 - Math.floor((shade + 1) * shadeIncrement);
            return color;
        });
    });
}

const CONFIG = {
    animation: {
        // Delay in ms between rendering each frame
        drawDelayMS: 1000.0 / 400.0,
        SMCycleLabelSize: 20,
        kernelHighlightPalette: generatePaletteRGBA(4),
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
        slotSize: 14,
        slotFillRGBA: [100, 100, 100, 0.15],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 15,
    },
    cache: {
        // Size of a L2 cacheline in slots
        L2CacheLineSize: 8,
        L2CacheLines: 16,
        cachedStateRGBA: [60, 60, 240, 0.5],
        pendingStateRGBA: [60, 60, 240, 0.2],
    },
    SM: {
        count: 2,
        warpSize: 32,
        warpSchedulers: 2,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 1,
        paddingX: 20,
        paddingY: 20,
        height: 105,
        colorRGBA: [100, 100, 100, 0.8],
    },
    grid: {
        dimGrid: {
            x: Math.round(32/8),
            y: Math.round(32/8),
        },
        dimBlock: {
            x: 8,
            y: 8
        },
    },
};
