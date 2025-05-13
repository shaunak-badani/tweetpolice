import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";


const Mean = () => {

    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState({});

    const presetValues = [
        "bitches get cut off everyday B ",
        "...Son of a bitch took my Tic Tacs."
    ]

    const handlePromptInput = async(query: string) => {
        setLoading(true);
        const response = await backendClient.get("/get-sentiment", {
            params: {
                input_text: query
            }
        });
        console.log(response.data);
        setResponse(response.data);
        setLoading(false);
    }

    return (
        <>
            <Textarea
                value={query}
                onChange = {(e) => setQuery(e.target.value)} 
                placeholder="Enter your tweet here!" />
            <div className="my-6">
                <Select onValueChange={setQuery}>
                    <SelectTrigger id="querySelector">
                    <SelectValue placeholder="Choose a sample tweet from the presets.." />
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
            {Object.entries(response).length > 0 && <HateBars prediction={response} />}
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};

type Prediction = Record<string, number>;

const HateBars = ({ prediction }: { prediction: Prediction }) => {
    return (
        <>
            {Object.entries(prediction).map(([key, value]) => 
                <div className="m-6">
                    <p className="text-lg">{key}: {value.toFixed(2)} %</p>
                    <Progress key={key} value={value} 
                        />
                </div>
                )
                
            }
        </>
    );
};


export default Mean;