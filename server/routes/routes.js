const controller = require("../controllers/controller.js");
try{
module.exports = (app) => {
    app.get("/RenalGuardian",controller.start);
    app.post("/RenalGuardian/predict",controller.predict);
    app.get("*",controller.error);
}
}
catch(err){
    console.log(err);
}