const mg = require('mongoose');
const ctoj = require('csvtojson');
const schema = require('./Schemas.js')
mg.connect("mongodb://localhost:27017/Project_RG");
const db = mg.connection;

    //collections of our database
  const pre_processed_data = mg.model("pre_processed_data",schema.pre_processed_schema);
  const raw_data = mg.model("raw_data",schema.raw_data_schema);

    //methods for inserting the data intp db
  exports.insert_raw_data = (data)=>
  {
    raw_data.insert(data).then((res)=>{
        console.log("inserted");
    }).catch((err)=>{
        console.log("error"+err);
    })
  }
  exports.insert_pre_processed_data = (data)=>
  {
    pre_processed_schema.insert(data).then((res)=>{
        console.log("inserted");
    }).catch((err)=>{
        console.log("error"+err);
    })
  }
  
//   ctoj().fromFile('D:\\coding\\Project_Ctrl-Alt-Defeat\\mongodb\\kidney_disease.csv')
//   .then(data =>{
//       console.log(data);
//       raw_data.insertMany(data).then(
//         ()=>{console.log("Inserted the data");}
//       ).catch((err)=>{
//         console.log("error occured"+err);
//       })
//   }).catch((err)=>{
//   console.log(err);
//   });
// let user = mg.model("user",userSchema);
// user.create({
//     name : "Eshwar",
//     age :20,
//     email :"eshwarvengala30@gmail.com"
// })