import { Button } from "@/components/ui/button";
import backendClient from "@/backendClient";

const Traditional = () => {

    const handlePromptInput = async() => {
        const response = await backendClient.get("/error");
    }

    return (
        <>
            <h6 className="pb-6 sm:pb-6 text-xl">Traditional model</h6>
            <Button variant="destructive" className="p-6 sm:p-6 rounded-2xl m-8 sm:m-8" onClick={handlePromptInput}>
                Simulate error
            </Button>
        </>
    )
};


export default Traditional;