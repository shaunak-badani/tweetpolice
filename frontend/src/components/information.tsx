import useGlobalStore from "@/store/store";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Info } from "lucide-react"

const Information = () => {

    const information = useGlobalStore(state => state.information);
    
    return (
        <div className="my-6 sm:my-6 py-2 sm:py-2">
        {   information.map(info => (

                <Card>
                    <CardHeader className="sm:text-sm">
                        <CardTitle className="flex flex-row">
                            <div className="mx-3 my-auto">
                                <Info className="text-blue-500"/>
                            </div>
                            <div>
                                {info}
                            </div>
                        </CardTitle>
                    </CardHeader>
                </Card>
                ))
        }
        </div>
    );
}

export default Information;