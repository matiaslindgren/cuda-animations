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
                [47, 21, 182, alpha],
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
        none: {
            name: "No",
            arithmetic: 0,
            L2CacheAccess: 0,
            memoryAccess: 0,
        },
        veryLow: {
            name: "(memory access only) Very low",
            arithmetic: 0,
            L2CacheAccess: 0,
            memoryAccess: 5,
        },
        low: {
            name: "Low",
            arithmetic: 0,
            L2CacheAccess: 0,
            memoryAccess: 25,
        },
        realisticLow: {
            name: "Realistic low",
            arithmetic: 1,
            L2CacheAccess: 20,
            memoryAccess: 70,
        },
    },
    memory: {
        // Empty space between each slot on all sides
        slotPadding: 1,
        slotSize: 16,
        slotFillRGBA: [160, 160, 160, 0.1],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 10,
    },
    cache: {
        // Size of a L2 cacheline in slots
        L2CacheLineSize: 8,
        L2CacheLines: {
            min: 0,
            max: 8*8,
            increment: 8,
        },
        cachedStateRGBA: [120, 120, 120, 0.4],
        pendingStateRGBA: [120, 120, 120, 0.2],
    },
    SM: {
        count: {
            min: 1,
            max: 4,
        },
        warpSize: 32,
        warpSchedulers: 1,
        // Probability of skipping an SM cycle (for simulating hardware latency)
        // Setting this to 0 makes all SMs to execute in unison
        latencyNoiseProb: 0.05,
    },
};
