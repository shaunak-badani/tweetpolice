import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";


const Mean = () => {

    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");

    const presetValues = [
        "What is the weather in New york today?",
        "What is 2 + 2?"
    ]

    const handlePromptInput = async(query: string) => {
        setLoading(true);
        const response = await backendClient.get("/mean", {
            params: {
                query: query
            }
        });
        setResponse(response.data.message);
        setLoading(false);
    }

    return (
        <>
            <h6 className="pb-6 sm:pb-6 text-xl">Mean model</h6>
            <Textarea
                value={query}
                onChange = {(e) => setQuery(e.target.value)} 
                placeholder="Enter your query here!" />
            <div className="my-6">
                <Select onValueChange={setQuery}>
                    <SelectTrigger id="querySelector">
                    <SelectValue placeholder="Choose a sample query from the presets.." />
                    </SelectTrigger>
                    <SelectContent position="popper">
                    {presetValues.map((preset, index) => 
                        (<SelectItem
                            key={index}
                            value={preset} 
                            >
                            {preset}
                        </SelectItem>)
                    )}
                    </SelectContent>
                </Select>
            </div>
            <Button className="p-6 sm:p-6 rounded-2xl m-8 sm:m-8" onClick={() => handlePromptInput(query)}>
                Send
            </Button>
            {response.length > 0 && <p>{response}</p>}
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};


export default Mean;