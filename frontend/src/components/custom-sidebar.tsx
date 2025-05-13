import { cn } from "@/lib/utils";
import { Card } from "./ui/card";
import useGlobalStore from "../store/store"
import { log } from "console";

export default function Sidebar(props:any) {

  const information = useGlobalStore(state => state.information)
  const user = useGlobalStore(state => state.user);
  log("user : ", user);

  const {children} = props;
  return (
    <div className="flex">
      <Card
        className={cn(
          "p-4 w-64",
        )}
      >
        <div className="flex justify-center items-center mb-4">
          <div className="text-xl font-semibold my-4">Sidebar</div>
        </div>
        
        <div>
        {information.map(info => 
          (<Card className="p-6 sm:p-6">{info}</Card>)
        )}
        </div>
      </Card>

      <div className="flex-1 p-6">
          {children}
      </div>
    </div>
  );
}