"use strict";

function failHard() {
    drawing = false;
    const errorBanner = document.getElementById("body-error-banner");
    errorBanner.innerHTML = "Something went wrong, please see the developer console";
    errorBanner.hidden = false;
    cancelDraw();
}

function assert(expr, msg, state) {
    if (!expr) {
        failHard();
        console.error("ASSERTION FAILED");
        if (typeof state !== "undefined") {
            console.error(state.name + " was:");
            printobj(state.obj);
        }
        throw "AssertionError: " + msg;
    }
}

function printobj(o) {
    console.log(JSON.stringify(o, null, 2));
}


function get4Palette(key) {
    switch(key) {
        case "rgba-colorful":
            const alpha = 0.10;
            return [
                [35, 196, 1, alpha],
                [47, 21, 162, alpha],
                [227, 1, 23, alpha],
                [235, 190, 2, alpha],
            ];
        default:
            failHard();
            console.error("unknown palette key: " + key);
    }
}


const CONFIG = {
    animation: {
        // Array of colors for highlighting kernel source lines and SMs
        kernelHighlightPalette: get4Palette("rgba-colorful"),
    },
    // Simulated latency in cycles for different instruction types
    latencies: {
        arithmetic: 2,
        L2CacheAccess: 4,
        memoryAccess: 16,
    },
    memory: {
        // Empty space between each slot on all sides
        slotPadding: 1,
        slotSize: 16,
        slotFillRGBA: [150, 150, 150, 0.1],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 8,
    },
    cache: {
        // Size of a L2 cacheline in slots
        L2CacheLineSize: 8,
        L2CacheLines: {
            default: 5*8,
            min: 1*8,
            max: 8*8,
            increment: 8,
        },
        cachedStateRGBA: [10, 10, 200, 0.2],
        pendingStateRGBA: [10, 10, 200, 0.1],
    },
    SM: {
        count: {
            default: 2,
            min: 1,
            max: 4,
        },
        warpSize: 32,
        warpSchedulers: 2,
        // Probability of skipping an SM cycle (for simulating hardware latency)
        // Setting this to 0 makes all SMs to execute in unison
        latencyNoiseProb: 0.05,
    },
};
