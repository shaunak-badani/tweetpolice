import { Backdrop } from "./backdrop";
import { Spinner } from "./spinner";

const BackdropWithSpinner = () => (
    <div>
        <Backdrop 
            open={true} 
            variant="blur" >
            <div className="animate-pulse">
                <Spinner size="medium" />
            </div>  
        </Backdrop> 
    </div>
);

export default BackdropWithSpinner;