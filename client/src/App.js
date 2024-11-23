import React, {useState, useEffect} from 'react';
import axios from 'axios';
function App(){

  const [word, setWord] = useState("not exited");
  const handleTurnOnCamera = () => {
    axios.post('http://localhost:3001/showcamera')
    .then(response => {
      console.log('Camera response:', response.data);
    })
    .catch(error => {
      console.error('There was error turning on the camera!', error);
    });
  };
  return (
      <div>
        <p>{word}</p>
        <button onClick={()=>setWord("exited")}>Click to exit</button>
        <div className='App'>
          <h1>React + Flask Camera App</h1>
          <button onClick={handleTurnOnCamera}>
            Turn On camera
          </button>
        </div>

      </div>
  )


}
export default App