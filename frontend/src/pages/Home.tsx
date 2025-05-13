import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Information from '../components/information'
import Mean from '../model-cards/mean'
import Traditional from '../model-cards/traditional'
import { Card } from "@/components/ui/card"



function Home() {
    return (
        <div>
              <h1 className="scroll-m-20 tracking-tight lg:text-3xl">
                Super catchy tagline!
              </h1>
              <p className="leading-7 [&:not(:first-child)]:mt-6 m-6 sm:m-6">
                Once upon a time, in a far-off land, there was a very lazy king who
                spent all day lounging on his throne. One day, his advisors came to him
                with a problem: the kingdom was running out of money (insert description)
              </p>
              <Information />
              <Tabs defaultValue="mean">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="mean">Mean model</TabsTrigger>
                  <TabsTrigger value="traditional">Traditional Model</TabsTrigger>
                </TabsList>
                <TabsContent value="mean">
                  <Card className="p-20">
                    <Mean />
                  </Card>
                </TabsContent>
                <TabsContent value="traditional">
                  <Card className="p-20">
                    <Traditional />
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
    );
  }
  
  export default Home;