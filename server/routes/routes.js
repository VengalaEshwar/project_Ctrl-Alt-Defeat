const controller = require("../controllers/controller.js");
module.exports = (app) => {
    app.get("/RenalGuardian",controller.start);
    app.post("/RenalGuardian/predict",controller.pedict);
    app.get("*",controller.error);
}