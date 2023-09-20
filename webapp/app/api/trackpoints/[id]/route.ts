import connectDB from "@/lib/connect-db"
import { createErrorResponse } from "@/lib/utils"
import { NextResponse } from "next/server"
import { getTrackpoint } from "../../../../lib/geolife-db"

export async function GET(
  _request: Request,
  { params }: { params: { id: string } }
) {
  try {
    await connectDB()

    const id = params.id
    const { trackpoint, error } = await getTrackpoint(id)

    if (error) {
      throw error
    }

    let json_response = {
      status: "success",
      data: {
        trackpoint,
      },
    }
    return NextResponse.json(json_response)
  } catch (error: any) {
    if (typeof error === "string" && error.includes("Trackpoint not found")) {
      return createErrorResponse("trackpoint not found", 404)
    }

    return createErrorResponse(error.message, 500)
  }
}
