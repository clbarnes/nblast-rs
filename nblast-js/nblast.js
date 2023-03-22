import { wrap } from "comlink";

const instance = new Worker(new URL("./nblast.worker.js", import.meta.url));
const Wrapper = wrap(instance);

export const getNblast = async () => { await new Wrapper() };

// export getNblast();
