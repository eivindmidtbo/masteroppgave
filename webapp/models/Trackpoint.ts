import {
  ModelOptions,
  Severity,
  getModelForClass,
  index,
  post,
  prop,
} from "@typegoose/typegoose"
import mongoose from "mongoose"

type GeoPoint = {
  type: string
  coordinates: number[]
}

@post<TrackpointClass>("save", function (doc) {
  if (doc) {
    doc.id = doc._id.toString()
    doc._id = doc.id
  }
})
@post<TrackpointClass[]>(/^find/, function (docs) {
  // @ts-ignore
  if (this.op === "find") {
    docs.forEach((doc) => {
      doc.id = doc._id.toString()
      doc._id = doc.id
    })
  }
})
@ModelOptions({
  schemaOptions: {
    timestamps: true,
    collection: "trackpoint",
  },
  options: {
    allowMixed: Severity.ALLOW,
  },
})
@index({ activity_id: 1 })
class TrackpointClass {
  @prop({ required: true, unique: true })
  activity_id: number

  @prop({ required: true, unique: false })
  location: GeoPoint

  @prop({ required: true, unique: false })
  altitude: number

  @prop({ required: true, unique: false })
  date_time: Date

  _id: mongoose.Types.ObjectId | string

  id: string
}

const Trackpoint = getModelForClass(TrackpointClass)
// const Trackpoint = mongoose.model("trackpoint")
export { Trackpoint, TrackpointClass }
