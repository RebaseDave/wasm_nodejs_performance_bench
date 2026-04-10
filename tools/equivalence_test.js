const jsPure = require("../src/variants/js_pure");
const jsWasm = require("../src/variants/js_wasm");
const jsWasmSimd = require("../src/variants/js_wasm_simd");
const jsWorker = require("../src/variants/js_worker");
const jsWorkerWasm = require("../src/variants/js_worker_wasm");
const jsWorkerWasmSimd = require("../src/variants/js_worker_wasm_simd");
const { TestImageGenerator } = require("./test-image-generator");

//checks if output buffers are equivalent (0.1% pixel difference for float rounding)
function arraysEqual(a, b) {
    if (a.length !== b.length) {
        return false;
    }
    let diff = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            diff++;
        }
    }
    console.log(`Differences: ${diff}/${a.length} pixels (${(diff / a.length * 100).toFixed(4)}%)`);
    return diff < a.length * 0.001;
}

async function test() {

    //equivalencetest: variants must produce the same output for identical inputs
    const base = TestImageGenerator.generate(100, 100, 42);

    //copies from base buffer
    const i1 = base.slice();
    const i2 = base.slice();
    const i3 = base.slice();
    const i4 = base.slice();
    const i5 = base.slice();
    const i6 = base.slice();

    const r1 = await jsPure.run(i1, 100, 100, 5, { passes: 1 });
    const r2 = await jsWasm.run(i2, 100, 100, 5, { passes: 1 });
    const r3 = await jsWasmSimd.run(i5, 100, 100, 5, { passes: 1 });
    const r4 = await jsWorker.run(i3, 100, 100, 5, { size: 1, passes: 1 });
    const r5 = await jsWorkerWasm.run(i4, 100, 100, 5, { size: 1, passes: 1 });
    const r6 = await jsWorkerWasmSimd.run(i6, 100, 100, 5, { size: 1, passes: 1 });

    //unwrap result - variants return either Uint8Array or { out, timings }
    const o1 = r1.out || r1;
    const o2 = r2.out || r2;
    const o3 = r3.out || r3;
    const o4 = r4.out || r4;
    const o5 = r5.out || r5;
    const o6 = r6.out || r6;

    //compare variants against js_pure
    console.log("Comparing outputs:");
    console.log("Pure vs Wasm:", arraysEqual(o1, o2) ? "Match" : "Differ");
    console.log("Pure vs WasmSimd:", arraysEqual(o1, o3) ? "Match" : "Differ");
    console.log("Pure vs Worker:", arraysEqual(o1, o4) ? "Match" : "Differ");
    console.log("Pure vs WorkerWasm:", arraysEqual(o1, o5) ? "Match" : "Differ");
    console.log("Pure vs WorkerWasmSimd:", arraysEqual(o1, o6) ? "Match" : "Differ");

    //close workerpools
    await jsWorker.closeAll();
    await jsWorkerWasm.closeAll();
    await jsWorkerWasmSimd.closeAll();

    //passes scaling test
    const base2 = TestImageGenerator.generate(1024, 1024, 42);

    const i7 = base2.slice();
    const i8 = base2.slice();

    const r7 = await jsPure.run(i7, 1024, 1024, 15, { passes: 1, returnTimings: true });
    const r8 = await jsPure.run(i8, 1024, 1024, 15, { passes: 3, returnTimings: true });

    console.log("1 Pass:", r7.timings.computeMs, "ms");
    console.log("3 Passes:", r8.timings.computeMs, "ms");
    console.log("Ratio:", (r8.timings.computeMs / r7.timings.computeMs).toFixed(2), "x");
}

test().catch(console.error);