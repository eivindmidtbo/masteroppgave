// import {
//   ModelOptions,
//   Severity,
//   getModelForClass,
//   index,
//   post,
//   prop,
// } from "@typegoose/typegoose"
// import mongoose from "mongoose"

// @post<UserClass>("save", function (doc) {
//   if (doc) {
//     doc.id = doc._id.toString()
//     doc._id = doc.id
//   }
// })
// @post<UserClass[]>(/^find/, function (docs) {
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
//     collection: "user",
//   },
//   options: {
//     allowMixed: Severity.ALLOW,
//   },
// })
// @index({ title: 1 })
// class UserClass {
//   @prop({ required: true, unique: true })
//   has_labels: boolean

//   _id: mongoose.Types.ObjectId | string

//   id: string
// }

// const User = getModelForClass(UserClass)
// export { User, UserClass }
