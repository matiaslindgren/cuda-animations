const memoryCanvas = document.getElementById("memoryCanvas");
const memoryCtx = memoryCanvas.getContext("2d");

const SMCanvas = document.getElementById("SMCanvas");
const SMCtx = SMCanvas.getContext("2d");

function clear(ctx) {
    ctx.fillStyle = 'rgb(255, 255, 255)';
    ctx.fillRect(0, 0, memoryCanvas.width, memoryCanvas.height);
}

function drawRects(rects, ctx) {
    rects.forEach(rect => {
        if (typeof rect.fillStyle !== "undefined") {
            ctx.fillStyle = rect.fillStyle;
            ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        }
        if (typeof rect.strokeStyle !== "undefined") {
            ctx.strokeStyle = rect.strokeStyle;
            ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        }
    });
}

function createGrid() {
    const rowSlotCount = 30;
    const columnSlotCount = 30;
    const slotPadding = 2;
    const slotSize = Math.floor(Math.min(
        memoryCanvas.width/rowSlotCount - slotPadding,
        memoryCanvas.height/columnSlotCount - slotPadding,
    ));
    return Array.from(
        new Array(rowSlotCount * columnSlotCount),
        (slot, i) => {
            return {
                x: (i % rowSlotCount) * (slotSize + slotPadding),
                y: Math.floor(i / rowSlotCount) * (slotSize + slotPadding),
                width: slotSize,
                height: slotSize,
                fillStyle: "#eee",
            };
        }
    );
}

function draw(now) {
    clear(memoryCtx);
    clear(SMCtx);
    drawRects(slots, memoryCtx);
    window.requestAnimationFrame(draw);
}

const slots = createGrid();
window.requestAnimationFrame(draw);


