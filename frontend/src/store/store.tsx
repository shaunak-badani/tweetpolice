import { create } from 'zustand';

export type AppState = {
  user: String | null;
  setUser: (user: String | null) => void;
  information: String[];
  error: String | null;
  setError: (error: String | null) => void;
};

const useGlobalStore = create<AppState>((set) => ({
  user: "Jonah",
  setUser: (user) => set({ user }),
  information: ["This is a slow application. Please expect a delay of 10s - 2m in responses."],
  error: null,
  setError: (error) => set({ error })
}));

export default useGlobalStore;
export const globalStore = useGlobalStore;