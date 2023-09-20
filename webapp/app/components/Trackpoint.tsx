import { TrackpointClass } from "../../models/Trackpoint"

type TrackpointProps = {
  trackpoint: TrackpointClass
}

export default function Trackpoint({ trackpoint }: TrackpointProps) {
  console.log(trackpoint)
  return <></>
}
