const { Worker } = require("worker_threads");

class WorkerPool {
    constructor({ workerFile, size, enforceTransfer = true }) {
        if (typeof workerFile !== "string" || workerFile.length === 0) {
            throw new TypeError("workerFile must be non-empty string");
        }
        if (!Number.isInteger(size) || size <= 0) {
            throw new RangeError("size must be positive integer");
        }

        this.workerFile = workerFile;
        this.size = size;
        this.enforceTransfer = enforceTransfer;

        this.workers = [];
        this.idle = [];
        this.queue = [];
        this.nextJobId = 1;
        this.pending = new Map();

        this._init();
    }
    //spawn workers and register event handlers
    _init() {
        for (let i = 0; i < this.size; i++) {
            const w = new Worker(this.workerFile);
            w.on("message", (msg) => this._onMessage(w, msg));
            w.on("error", (err) => this._onError(err));
            w.on("exit", (code) => this._onExit(code));

            this.workers.push(w);
            this.idle.push(w);
        }
    }
// resolve or reject pending job, return worker to idle-pool
    _onMessage(worker, msg) {
        const jobId = msg && msg.jobId;
        const waiter = this.pending.get(jobId);
        if (!waiter) return;

        this.pending.delete(jobId);

        this.idle.push(worker);
        this._drain();

        if (!msg.ok) {
            waiter.reject(new Error(msg.error || "Worker job failed"));
            return;
        }

        const output = msg.out;
        //enforce that output was transferred per transferList from worker
        if (this.enforceTransfer) {
            if (!(output instanceof ArrayBuffer)) {
                waiter.reject(new TypeError("Worker output must be of type: ArrayBuffer"));
                return;
            }
        }
        const outU8 = new Uint8Array(output);
        //attach timings
        if (msg.timings && typeof msg.timings === "object") {
            waiter.resolve({ out: outU8, timings: msg.timings });
        }
        else {
            waiter.resolve(outU8);
        }
    }
    //reject pendng jobs on worker error
    _onError(err) {
        for (const [, waiter] of this.pending.entries()) {
            waiter.reject(err);
        }
        this.pending.clear();
    }

    _onExit(code) {
        if (code === 0) return;
        const err = new Error(`Worker exited with code ${code}`);
        this._onError(err);
    }
    //dispatch queued jobs to idle workers
    _drain() {
        while (this.idle.length > 0 && this.queue.length > 0) {
            const worker = this.idle.pop();
            const job = this.queue.shift();

            if (this.enforceTransfer) {
                if (!Array.isArray(job.transferList) || job.transferList.length === 0) {
                    throw new Error("TransferList must be provided for pixel buffer (zero-copy)");
                }
            }
            //calculate job queuetime
            if (job.msg && typeof job.msg === "object" && typeof job.msg._enqueueNs === "bigint") {
                const nowNs = process.hrtime.bigint();
                job.msg._dispatchDelayMs = Number(nowNs - job.msg._enqueueNs) / 1e6;
            }
            worker.postMessage(job.msg, job.transferList);
        }
    }
    //enqueue a job and return a promise that resolves with result
    run(payload, transferList) {
        if (!payload || typeof payload !== "object") {
            throw new TypeError("payload must be of type: object");
        }

        if (!(payload.input instanceof ArrayBuffer)) {
            throw new TypeError("payload.input must be of type: ArrayBuffer");
        }

        const jobId = this.nextJobId++;
        payload.jobId = jobId;
        payload._enqueueNs = process.hrtime.bigint();

        const tl = transferList || [payload.input];

        if (this.enforceTransfer) {
            if (!Array.isArray(tl) || tl.length === 0) {
                throw new Error("TransferList must be provided (zero-copy)");
            }
            const hasInput = tl.includes(payload.input);
            if (!hasInput) {
                throw new Error("TransferList must include payload.input ArrayBuffer");
            }
        }

        return new Promise((resolve, reject) => {
            this.pending.set(jobId, { resolve, reject });
            this.queue.push({ msg: payload, transferList: tl });
            this._drain();
        });
    }
    //terminate all workers and clear state
    async close() {
        const terms = this.workers.map((w) => w.terminate());
        await Promise.all(terms);
        this.workers = [];
        this.idle = [];
    }
}

module.exports = WorkerPool;