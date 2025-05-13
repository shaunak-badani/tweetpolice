import './App.css'
import { Card } from "@/components/ui/card"
// import { SidebarProvider } from './components/ui/sidebar'
// import Sidebar from './components/custom-sidebar'
import useGlobalStore from './store/store'
import ErrorOverlay from './model-cards/error-overlay'
import { NavigationMenuDemo } from './components/nav-menu'
import { Routes, Route } from "react-router-dom";
import Home from './pages/Home'
import { BrowserRouter } from "react-router-dom";
import Examples from './pages/Examples'

function App() {
  const error = useGlobalStore(state => state.error);
  console.log("Error: ", error);

  return (
    <BrowserRouter>
    <>
    <div>
      {/* <SidebarProvider> */}
      {/* <Sidebar> */}

      <ErrorOverlay />
      
        <Card className="grid grid-cols-3 grid-gap-4">
          <div className="col-span-1" />
          <div className="header p-6 text-4xl font-extrabold lg:text-4xl">
            AppName
          </div>
          <div className="col-span-1 flex items-center">
            <NavigationMenuDemo />
          </div>
        </Card>
        <div className="min-h-screen p-8 pb-8 sm:p-8">      
          <main className="max-w-4xl mx-auto flex flex-col gap-16">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/examples" element={<Examples />} />
            {/* Add a catch-all for 404 */}
            <Route path="*" element={<h2>404 - Page Not Found</h2>} />
          </Routes>
          </main>
        </div>
    {/* </Sidebar> */}
    {/* </SidebarProvider> */}
    </div>
    
      
    </>
    </BrowserRouter>
  )
}

export default App
