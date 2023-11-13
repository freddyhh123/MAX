
CREATE TABLE `tracks` (
  `track_id` varchar(30) PRIMARY KEY,
  `track_name` varchar(255),
  `preview_url` varchar(255),
  `spotify_url` varchar(255),
  `featureset_id` varchar(30),
  `time_added` timestamp,
  `trained` boolean,
  `batch` varchar(50)
);

CREATE TABLE `features` (
  `featureset_id` varchar(30) PRIMARY KEY,
  `danceability` float,
  `energy` float,
  `key` float,
  `loudness` float,
  `mode` float,
  `speechiness` float,
  `acousticness` float,
  `instrumentalness` float,
  `liveleness` float,
  `valence` float,
  `tempo` float,
  `duration` float
);
ALTER TABLE `tracks` ADD FOREIGN KEY (`featureset_id`) REFERENCES `features` (`featureset_id`);

CREATE TABLE `artists` (
  `artist_id` varchar(30) PRIMARY KEY,
  `artist_name` varchar(255),
  `artist_url` varchar(255)
);

CREATE TABLE `genres` (
  `genre_id` integer PRIMARY KEY AUTO_INCREMENT,
  `genre_name` varchar(255)
);


CREATE TABLE `albums` (
  `album_id` varchar(30) PRIMARY KEY,
  `album_name` varchar(255),
  `spotify_url` varchar(255)
);

CREATE TABLE `album_images` (
  `image_id` integer PRIMARY KEY AUTO_INCREMENT,
  `image_url` varchar(255),
  `image_height` integer,
  `image_width` integer
);


CREATE TABLE `track_genres` (
  `track_id` varchar(30),
  `genre_id` integer
);
ALTER TABLE `track_genres` ADD FOREIGN KEY (`genre_id`) REFERENCES `genres` (`genre_id`);
ALTER TABLE `track_genres` ADD FOREIGN KEY (`track_id`) REFERENCES `tracks` (`track_id`);


CREATE TABLE `artists_tracks` (
  `artist_id` varchar(30),
  `track_id` varchar(30)
);
ALTER TABLE `artists_tracks` ADD FOREIGN KEY (`artist_id`) REFERENCES `artists` (`artist_id`);
ALTER TABLE `artists_tracks` ADD FOREIGN KEY (`track_id`) REFERENCES `tracks` (`track_id`);


CREATE TABLE `artists_albums` (
  `artist_id` varchar(30),
  `album_id` varchar(30)
);
ALTER TABLE `artists_albums` ADD FOREIGN KEY (`artist_id`) REFERENCES `artists` (`artist_id`);
ALTER TABLE `artists_albums` ADD FOREIGN KEY (`album_id`) REFERENCES `albums` (`album_id`);


CREATE TABLE `album_tracks` (
  `track_id` varchar(30),
  `album_id` varchar(30)
);
ALTER TABLE `album_tracks` ADD FOREIGN KEY (`track_id`) REFERENCES `tracks` (`track_id`);
ALTER TABLE `album_tracks` ADD FOREIGN KEY (`album_id`) REFERENCES `albums` (`album_id`);

CREATE TABLE `image_links` (
  `album_id` varchar(30),
  `image_id` integer
);
ALTER TABLE `image_links` ADD FOREIGN KEY (`album_id`) REFERENCES `albums` (`album_id`);
ALTER TABLE `image_links` ADD FOREIGN KEY (`image_id`) REFERENCES `album_images` (`image_id`);