import './App.css'
import { Card } from "@/components/ui/card"
// import { SidebarProvider } from './components/ui/sidebar'
// import Sidebar from './components/custom-sidebar'
import useGlobalStore from './store/store'
import ErrorOverlay from './model-cards/error-overlay'
import Home from './pages/Home'

function App() {
  const error = useGlobalStore(state => state.error);
  console.log("Error: ", error);

  return (
    <>
    <div>
      <ErrorOverlay />
        <Card className="grid grid-cols-3 grid-gap-4">
          <div className="col-span-1" />
          <div className="flex justify-center p-6 text-3xl font-extrabold lg:text-4xl">
            TweetPolice
          </div>
        </Card>
        <div className="min-h-screen lg:p-8 lg:pb-8 p-4">      
          <main className="max-w-4xl mx-auto flex flex-col gap-16">
            <Home />
          </main>
        </div>
    </div>
    </>
  )
}

export default App
