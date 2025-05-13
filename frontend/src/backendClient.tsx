import axios from "axios";
import { log } from "./lib/logger"
import { globalStore } from "./store/store";

const apiURL = import.meta.env.VITE_API_ENDPOINT;
log("apiURL : ", apiURL);

const backendClient = axios.create({
    baseURL: apiURL
});

backendClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (!error.response) {
            console.log("Error here : ", error)
            // Backend unreachable (network error, CORS, DNS failure, etc.)
            let msg = "Cannot connect to backend server."
            globalStore.getState().setError(msg);
            return Promise.resolve({ data: null, error: { msg } });
        }
        const message =
            error.response?.data?.detail || "Something went wrong with the API.";
        const status = error.response?.status;

        // log("API error:", { status, message });
        
        globalStore.getState().setError(message);
        return Promise.resolve({ data: null, error: { message } });
    }
  );

export default backendClient;