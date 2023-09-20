import Image from "next/image"
import Trackpoint from "../components/Trackpoint"
import { getTrackpoints } from "../../lib/geolife-db"

export default async function Home() {
  const { trackpoints } = await getTrackpoints({ limit: 100 })
  console.log(trackpoints)
  return (
    <div>
      {trackpoints?.map((trackpoint) => (
        <Trackpoint key={trackpoint.id} trackpoint={trackpoint} />
      ))}
    </div>
  )
}
