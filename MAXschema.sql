DROP SCHEMA IF EXISTS max;
CREATE SCHEMA max;
USE max;

CREATE TABLE `tracks` (
  `track_id` varchar(30) PRIMARY KEY,
  `track_name` varchar(255),
  `file_path` varchar(255),
  `time_added` timestamp,
  `trained` boolean,
  `batch` varchar(50)
);

CREATE TABLE `features` (
  `featureset_id` varchar(30) PRIMARY KEY,
  `danceability` float,
  `energy` float,
  `speechiness` float,
  `acousticness` float,
  `instrumentalness` float,
  `liveleness` float,
  `valence` float,
  `tempo` float
);

CREATE TABLE `artists` (
  `artist_id` varchar(30) PRIMARY KEY,
  `artist_name` varchar(255),
  `artist_url` varchar(255)
);

CREATE TABLE `genres` (
  `genre_id` integer PRIMARY KEY,
  `genre_parent` integer,
  `top_genre` integer,
  `genre_name` varchar(255)
);

CREATE TABLE `albums` (
  `album_id` varchar(30) PRIMARY KEY,
  `album_link` varchar(255),
  `album_name` varchar(255)
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