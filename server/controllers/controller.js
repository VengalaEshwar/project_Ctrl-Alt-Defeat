const axios = require('axios');
exports.start =(req,res)=>{
    res.send("Server is Started");
}
exports.error =(req,res)=> {
    res.send("The is incorrect");
}
exports.postData =  async (req, res) => {
    try {
        // Send request to Python Flask API
        const response = await axios.post('http://localhost:5001/predict', req.body);
        const prediction = response.data;
        res.json(prediction);
    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ error: 'Internal server error' });
    }
}
//(req,res)=>{
//     const formData = req.body;
//   console.log('Received form data:', formData);
//   const predictedData = 
//   res.json({ message: 'Form data received successfully!' });

//}