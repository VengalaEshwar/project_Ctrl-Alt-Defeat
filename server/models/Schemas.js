exports.raw_data_schema = new mg.Schema({
    id: { type: Number },
  age: { type: Number },
  bp: { type: Number },
  sg: { type: Number },
  al: { type: Number },
  su: { type: Number },
  rbc: { type: String },
  pc: { type: String },
  pcc: { type: String },
  ba: { type: String },
  bgr: { type: Number },
  bu: { type: Number },
  sc: { type: Number },
  sod: { type: Number },
  pot: { type: Number },
  hemo: { type: Number },
  pcv: { type: String },
  wc: { type: String },
  rc: { type: String },
  htn: { type: String },
  ane: { type: String },
  classification: { type: String }
  });
exports.pre_processed_schema = new mg.Schema(
    {
        id: { type: Number },
        age: { type: Number },
        bp: { type: Number },
        sg: { type: Number },
        al: { type: Number },
        su: { type: Number },
        rbc: { type: Number },
        pc: { type: Number },
        pcc: { type: Number },
        ba: { type: Number },
        bgr: { type: Number },
        bu: { type: Number },
        sc: { type: Number },
        sod: { type: Number },
        pot: { type: Number },
        hemo: { type: Number },
        pcv: { type: Number },
        wc: { type: Number },
        rc: { type: Number },
        htn: { type: Number },
        ane: { type: Number },
        classification: { type: Number }
      }
  );
exports.userSchema = mg.Schema({
    name : String,
    age  : Number,
    email : String
});
  