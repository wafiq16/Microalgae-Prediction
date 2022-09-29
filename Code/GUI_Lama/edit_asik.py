height_final_begin = int((self.begin.y() - 400)/320 * self.imageCamNow.shape[0])
                width_final_begin = int((self.begin.x() - 50)/480 * self.imageCamNow.shape[1])

                height_final_end = int((self.end.y() - 350)/320 * self.imageCamNow.shape[0])
                width_final_end = int((self.end.x() - 50)/480 * self.imageCamNow.shape[1])

                if width_final_end > self.imageCamNow.shape[1]:
                    width_final_end = self.imageCamNow.shape[1]
                if height_final_end > self.imageCamNow.shape[0]:
                    height_final_end = self.imageCamNow.shape[0]

                self.imageCamNow = self.imageCamNow[height_final_begin:height_final_end,width_final_begin : width_final_end]