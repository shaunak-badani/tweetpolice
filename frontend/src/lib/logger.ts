export const log = (...args: any[]) => {
    if (!import.meta.env.PROD) {
        console.log(...args);
    }
};