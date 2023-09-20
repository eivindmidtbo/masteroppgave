// import {
//   ModelOptions,
//   Severity,
//   getModelForClass,
//   index,
//   post,
//   prop,
// } from "@typegoose/typegoose"
// import mongoose from "mongoose"

// @post<ActivityClass>("save", function (doc) {
//   if (doc) {
//     doc.id = doc._id.toString()
//     doc._id = doc.id
//   }
// })
// @post<ActivityClass[]>(/^find/, function (docs) {
//   // @ts-ignore
//   if (this.op === "find") {
//     docs.forEach((doc) => {
//       doc.id = doc._id.toString()
//       doc._id = doc.id
//     })
//   }
// })
// @ModelOptions({
//   schemaOptions: {
//     timestamps: true,
//     collection: "activity",
//   },
//   options: {
//     allowMixed: Severity.ALLOW,
//   },
// })
// @index({ title: 1 })
// class ActivityClass {
//   @prop({ required: true, unique: true })
//   user_id: string
//   @prop({ required: true, unique: false })
//   transportation_mode: string | boolean
//   @prop({ required: true, unique: false })
//   start_date_time: Date
//   @prop({ required: true, unique: false })
//   end_date_time: Date

//   _id: mongoose.Types.ObjectId | string

//   id: string
// }

// const Activity = getModelForClass(ActivityClass)
// export { Activity, ActivityClass }
