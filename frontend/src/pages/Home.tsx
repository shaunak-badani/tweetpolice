import Information from '../components/information'
import Mean from '../model-cards/mean'


function Home() {
    return (
        <div>
              <h1 className="scroll-m-20 tracking-tight lg:text-3xl text-2xl">
                Find out if your tweet is hateful, offensive or neither!
              </h1>
              <p className="leading-7 [&:not(:first-child)]:mt-6 lg:m-6 sm:m-2">
                TweetPolice helps you identify if your tweet is hateful, offensive or neither. It does this by passing your tweet through a BERT model that has been fine tuned on some tweets of the same nature, and returning the predictions of the same.
              </p>
              <Information />
              <Mean />
            </div>
    );
  }
  
  export default Home;