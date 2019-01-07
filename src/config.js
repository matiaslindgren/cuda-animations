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

function dumpCachelines(o) {
    for (let [index, line] of o.entries()) {
        console.log("index", index, "width", line.lineSize);
        console.log("  " + Array.from(line.indexes).join(' '));
    }
}

function disjoint(set1, set2) {
    for (let x of set1.values())
        if (set2.has(x))
            return false;
    return true;
}

function get4Palette(key) {
    switch(key) {
        case "rgba-colorful":
            const alpha = 0.2;
            return [
                [35, 196, 1, alpha],
                [227, 1, 23, alpha],
                [235, 190, 2, alpha],
                [47, 21, 182, alpha],
            ];
        default:
            failHard();
            console.error("unknown palette key: " + key);
    }
}


const CONFIG = {
    animation: {
        // Array of distinct colors to distinguish independent streaming multiprocessors
        SMColorPalette: get4Palette("rgba-colorful"),
    },
    // Simulated latency in cycles for different instruction types
    // Mei and Chu [1] report that global memory access latencies on a GTX980 can be measured in tens if there is an L1 TLB hit, while on an L1 TLB miss, the latencies go up to hundreds or thousands of cycles.
    latencies: {
        none: {
            name: "No",
            arithmetic: 0,
            L2CacheAccess: 0,
            memoryAccess: 0,
        },
        veryLow: {
            name: "(DRAM access only) Very low",
            arithmetic: 0,
            L2CacheAccess: 0,
            memoryAccess: 5,
        },
        medium: {
            name: "Medium",
            arithmetic: 0,
            L2CacheAccess: 2,
            memoryAccess: 20,
        },
        high: {
            name: "High",
            arithmetic: 1,
            L2CacheAccess: 5,
            memoryAccess: 50,
        },
        realistic: {
            name: "Realistic",
            arithmetic: 10,
            L2CacheAccess: 50,
            memoryAccess: 500,
        },
    },
    memory: {
        // Empty space between each slot on all sides
        slotPadding: 1,
        slotSizes: {
            min: 8,
            max: 20,
            step: 4,
        },
        slotFillRGBA: [160, 160, 160, 0.2],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 15,
    },
    cache: {
        // Size of a L2 cacheline in slots
        L2CacheLineSize: 8,
        L2CacheLines: 40,
        cachedStateRGBA: [160, 160, 160, 0.6],
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
