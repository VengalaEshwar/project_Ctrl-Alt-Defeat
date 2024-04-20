const model = require("../models/models.js");
const axios = require('axios');
exports.start =(req,res)=>{
    res.send("Server is Started");
}
exports.error =(req,res)=> {
    res.send("The is incorrect");
}
exports.predict =  async (req, res) => {
    try {
        console.log(req.body);
        //waiting for the responsrr from the python server
        const response = await axios.post('http://localhost:5001/predict', req.body);

        const prediction = response.data;

        //inserting the predicated raw and processed data into the database

        // let raw_data = req.body;
        // let processed_data = new Object(prediction);
        // raw_data['classification'] = prediction==1?"notckd":"ckd";
        // processed_data['classification']=prediction;
        // model.insert_pre_processed_data(processed_data);
        // model.insert_raw_data(raw_data);
        
        //suggestion from the prediction
        const result =  prediction//suggest(req.body,prediction);

        res.json(result);
    } catch (error) {
        console.error('Error:', error.message);
        res.status(500).json({ error: 'Internal server error' });
    }
}