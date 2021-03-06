CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??E????      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
8?   max       P??]      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??P   max       >%      ?  ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?z?G?   max       @E??????     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??Q???    max       @vo?z?H     	`  )?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @1?        max       @Q`           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @?`          ?  3?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?\)   max       >}??      ?  4?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??   max       B*?d      ?  5?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?
/   max       B*?4      ?  6?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =?GA   max       C?ie      ?  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =m??   max       C?h?      ?  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      ?  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      ?  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??   max       P?Mw      ?  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ????Y??}   max       ?????m\?      ?  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??P   max       >-V      ?  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?33333   max       @E?          	`  >?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??
=p??    max       @vo?z?H     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @-         max       @R`           x  Q?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?F        max       @???          ?  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B?   max         B?      ?  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ?z6??C-   max       ???-V     ?  T      	            .                         	            T                   <      -            *            "                     m   .   k   	      9            -   	         ?   ?               2N?bNx&?N(?7NW?+N?M%Px?RN???O3?9NkcaOS7?N?kN???O??NFs?N???NʤOg?P??]N???N??3N
8?N?][P)/>PA4N?(O\|?O5BN"?O??SP ?N|??N?BO?H?O???O/?@O?\N???N?V?O??O???P=wePyOP?X?Nu??O0??O?UzNN-4OD??O>ڰPA?mN?5<O???No"hPF??O?ӏN4>?O?~N??N2?ON???P?ě??ě????
%   ;D??<#?
<49X<D??<D??<T??<e`B<e`B<e`B<u<u<?t?<?t?<?t?<???<???<???<??
<??
<?1<?1<?9X<?j<ě?<?`B<?`B<?`B<???=+=C?=C?=C?=t?=t?=t?=?P=?P=?w=?w=<j=H?9=Y?=]/=]/=aG?=m?h=?%=?hs=???=Ƨ?=??=??=?
==??>%>BO[_\[OHB>>>>>>>>>>????????????????????b`bin{{{ywpnlbbbbbbb????????????????????????????????????????++1BOht?????????hT6+CEJNS[gjpmg\[NCCCCCC??????????|wz???????||||||||||????????????????????fgglpt?????????trkgf???????????????????????????
??????

??????????????????????????????????????????????????????????????????????????5SV]a`[N)????????????????33<HU^_acba`UHB<3333IOT[hnoh[OIIIIIIIIIIst~?????????}tssssss?????/<NOHBD<6#
???????
!%;AUXWH/#????????????????????????????????????????????VX[fhlt?????????th[V
????









???
???????????JENOS\l??????????cUJ????????????????????????????????????????????)5>:0/) ??./-0BFO]aakquwthYB6.??????
#*-#!
????7569;AHU]addba^UH><7????????????????????????	?????????????????????????????????#,)*%?????????)28CMNIB5)????/+,5BN[t??????tg[N5/?????? ????????????????????????????????)6;4)????

6CNSTQHB6)
?????????????????????	"/;HTZTMH;/"	 ??????????????????????????),)???????!"/4;><;4/"????????????????????zz??????????zzzzzzzzxwz???????????????~x??????????????aagnz???znaaaaaaaaaqnnnnpz??????????zqq

"#+*(#




`ZUZaflnnnma````````??????"##
???????????????????????????????????????????????????????????????????????????????????̻x?????????????????x?l?g?l?x?x?x?x?x?x?x?r?~??????????????????~?r?q?r?r?r?r?r?r???????????ù˹ϹҹϹù????????????????????????Ѿ׾Ծľ????????s?J?F?H?F?G?M?f???U?b?n?u?{ŀ?}?{?n?b?[?U?U?S?U?U?U?U?U?U?-?:?F?S?_?m?x?x?l?_?S?F?.?!??
???!?-?@?L?Y?_?d?Y?L?@?4?9?@?@?@?@?@?@?@?@?@?@???????"?3?7?4?.?'???????ݻܻػܻ????n?z?{ÇÓÛÓÇ?z?n?f?a?]?U?P?R?U?a?j?n?5?B?N?T?V?N?B?6?5?5?)?!?)?/?5?5?5?5?5?5?<?U?a?n?x?t?u?s?n?a?U?<?/?#????#?/?<?????????????????????????????????????????)?6???B?H?O?Q?O?B?@?6?*?)?!??%?)?)?)?)E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E????*?6?7?C?E?D?C?6?*????
?????ƎƧ???????????ƚ?u?]?R?N?Q?P?S?h?}Ǝ????????????????ùîìèìóù???????????????? ?????????????????????????????????H?M?U?Y?Y?U?H?D?D?D?H?H?H?H?H?H?H?H?H?H?<???H?I?M?H?B?<?/?&?%?,?/?;?<?<?<?<?<?<?H?T?a?p?s?l?\?J??	??????????????"?8?H???????????????y?m?G?;?.?!?$?&?;?m?|?????!?-?:?C?F?J?L?F?:?.?-?*?!? ?!?!?!?!?!?!ù??????????þìÙÓÍÇÃÀÇ×àìõù?-?:?;?F?L?S?Y?U?S?K?F?:?-?)?+?%?"?#?+?-?3?3?7?@?L?O?Y?e?Y?L?@?3?3?3?3?3?3?3?3?3?ּӼʼ???????????y?|??????????Ǽּ??????????"?/?;?I?L?C?;?/????????????????????????????????? ?????????????????????????????????????????????ҿѿݿ??????????????ݿοɿ??¿Ŀοѻлܻ?????)?'????????ܻջŻû??Ļ??????????????????????????????????????????)?2?4?)?#?????????????????zÆÀ?{?z?o?n?a?U?I?U?Y?a?k?n?w?z?z?z?z?a?m?z???z?z?m?g?a?T?K?M?T?_?a?a?a?a?a?a??????????????????????????????????????s????????ɾɾ?????????f?Z?4?1?;?M?Z?s?0?I?U?X?W?G?<?0??????ĿĴķľ????????0?B?h?tēĦġĕ?v?n?n?h?Y?6?)??#?4?7?7?Bàù?????????????ùÓ?z?a?E?&?/?dÇà?A?M?Z?f?h?g?f?Z?M?A?@???A?A?A?A?A?A?A?A???????w?u?u?s?m?f?Z?M?F?H?M?T?Z?f?s??r?~???????????ͺֺ????????~?q?b?\?^?f?r?A?N?Z?[?g?q?g?g?g?Z?N?D?A?;?A?A?A?A?A?A?a?i?m?q?x?~?????z?m?l?a?]?W?O?N?I?N?T?a???Ľнݽ????????ݽĽ??????|?????????????????????$?4?:?8?"?	?????????????????????
???#?%?%?#? ??
???????????
?
?
?
?????????ƽĽĽ½????????????????z?v?x???(?5?>?=?5?-?(????? ?(?(?(?(?(?(?(?(?f?r???????ʼ??????ּ????r?Y?K?C?E?M?Y?fD?D?D?D?D?D?D?D?D?D?D?D?D?D{DpDlDnD{D?D??????????????????????:?F?S?_?l?x???x?w?l?e?_?S?H?F?:?2?.?:?:??*?6?@?C?O?R?O?O?C?6?*?"????????????ûлջлû??????????????????????????{ŇŔŖşŝŔōŇ?{?n?b?U?O?V?_?h?n?y?{ Y , ? \ H A L m ; . J 7 5 8 ` 5 ( + d X x R ` g l 9 R V ; 2 D ; D + % 6 ? j $ \ M M ] 0 ? 8 j j j c # u X 4 C \ C 6 M C  N  t  S  ?  ?     ?  ?  r  ?    |  2  g  ?  9  V    ?  ?  m  ?  ?  J  ?  ?  r  U    ?  ?  ?  m  N  z  \  ?  ?  %  k  ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  \  ?  i  `  ?  B  ?  U  Ľ\)?u??o%   <e`B=D??<?/<?9X<?9X='??<?<?t?=8Q?<?j<???<?9X<?=??=o=o<ě?<???=T??=??
<???=?+=C?<?/=@?=?O?=\)=#?
=49X=?+=e`B=Y?=<j=,1=,1=m?h>z?=???>z?=D??=??=?
==u=??P=?hs=???=?+=???=???>}??>fff=?`B>%>   >o>49XB??B-B(L
BʬBg?B??B?XB??B?B!??B

?B??B"VB??B??B??B?B?;B??B#?B;?B@/B??B?AB!1:B"MB`?B$4?B"?>B
?BTuB??B??B??B??BzBk?BTIB?B?NB?dB	#?B)FBriB??Bi?BuA?#RB"??B?A??B*?dB
??BoB?B??Bi?BOJBIBwBc9B>?B(<xB??BBuB??B?PB:?B?-B!??B	?pB??B=?B??B?BL?B^?B?9B5?B5?BKBCqB?BL?B!I B"@B~BB$GB"?B
@?BB?B??B??BH?B??B?HB??BA?B??B??B?RB>>B?cB??B@?B=?B"A?x?B">bBиA?
/B*?4B
c?BA?B??B=?B?HB??B@?BA~A.¦A?|o@???@?=?GAAF?A?ޞ@?a??t@??	A???A?2AĘ$A?'A?m?C?ieA?/XB[A?Y?A??rAĚ?A?#?A?K?Ak?z@ym?A?e?@{????@?%?A? A???AЎ?A;@???A?? A??EA?SA??eAIL]AE??A??=A?9?A?g?A=??A@]?@-.A?yA??UA%M?A??A?̪A!zA???@鵵C???A2??@?w?B ,:@?A?QbA/?A?~c@??@?n=m??AE ?A?f?@n????=?@?
:Aǚ0A???A??A??A?n?C?h?A???B1?A?|?A?rHA?piA?A?PAki@{??À?@??6????@?2A??fA?~?A?o?A}	@?b<A???A??;AƋ?A?W?AInBAD??A??Aہ?A̫?A>?"A?@LRA??A??A'NA?k5A??`A?@A? ;@??C?קA3@@?B 2?@?0A???      
            .                         
            T               !   =      .            *            "                     m   .   k   
      9            .   	         ?   ?               3                  1                                    9               /   /                  '                              '   -   /   ?         !            3      #      /                                    -                                    7               -   #                  !                              %      )                        3                              N?bNx&?N(?7NW?+NjG_P4??N?l?O3?9N?XO&K?N?kN???O??NFs?N6??NʤOg?P?MwN???NU??N
8?NopPA+O?WONl??N?DvN??rN"?O^?O??>N|??N?BO??7O-??O˧N??3M??N?V?O??O?m?O?#Pe-O??#Nu??N??1O??7NN-4O5??O>ڰPA?mN?5<N??No"hO[YO*I?N4>?O?~N??N2?ON?  v  y      ?    g  ?  ?  ?  ?  /  P  ;  O  ?  ?  ?  \  T  ?  ?  
  ?  ?  (  ?  b  ?  p  ?        #  {  ?  Y  |  E  ?  ?  
    v  ?  ?      ?  ?  B  ?  )  ?  n  ?  }  ?  ??P?ě??ě????
:?o<D??<49X<49X<e`B<?o<T??<e`B<e`B<e`B<?C?<u<?t?<?1<?t?<?1<???<??
<?9X=?P<?9X=?P<???<?j<???=C?<?`B<?`B=o=0 ?=?P=?w=#?
=t?=t?=??=?t?=??=?-=?w=L??=e`B=Y?=aG?=]/=aG?=m?h=?\)=?hs>-V>
=q=??=??=?
==??>%>BO[_\[OHB>>>>>>>>>>????????????????????b`bin{{{ywpnlbbbbbbb????????????????????????????????????????48BO[t?????????h[E<4GFKNT[gipmg[[NGGGGGG???????????y{?????????????????????????????????????fgglpt?????????trkgf???????????????????????????
??????

?????????????????????????????????????????????????????????????????????????)5N\`_[NB)?????????????????55<HUUZUH<5555555555IOT[hnoh[OIIIIIIIIII~v??????????~~~~~~~~?????/<LNH></#
?????????
/5GKLH/#
??????????????????????????????????????????`_hhqt~???????th````
????









?????????
 ???????TQUX^gt?????????tg]T?????????????????????????????????????????????)598/-)?4368BGOT[_hinh[OB764????
"###
?????F>>@EHUYaaaa]UPHFFFF????????????????????????	?????????????????????????????????#)(#???????????)16;;5)??0+,5BN[t??????tg[N50??????????????????????????????????????????)/575/)????
)6BJNOMFB6)?????????????????????	"(/;HRTLH;/"	?????????????????????????),)???????!"/4;><;4/"????????????????????zz??????????zzzzzzzz???????????????????????????? 

??????aagnz???znaaaaaaaaaqnnnnpz??????????zqq

"#+*(#




`ZUZaflnnnma````````??????"##
???????????????????????????????????????????????????????????????????????????????????̻x?????????????????x?l?g?l?x?x?x?x?x?x?x?r?~??????????????????~?r?q?r?r?r?r?r?r?????ùʹϹѹϹù??????????????????????????????ǾǾ????????????s?X?P?N?O?T?f?s???U?b?n?t?{ŀ?|?{?n?b?\?U?U?T?U?U?U?U?U?U?-?:?F?S?_?m?x?x?l?_?S?F?.?!??
???!?-?@?L?Y?]?`?Y?L?@?????@?@?@?@?@?@?@?@?@?@?????????'?/?1?'????????????߻????n?z?{ÇÓÛÓÇ?z?n?f?a?]?U?P?R?U?a?j?n?5?B?N?T?V?N?B?6?5?5?)?!?)?/?5?5?5?5?5?5?<?U?a?n?x?t?u?s?n?a?U?<?/?#????#?/?<?????????????????????????????????????????)?6?=?B?D?L?B?B?6?)? ?'?)?)?)?)?)?)?)?)E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E????*?6?7?C?E?D?C?6?*????
?????Ƨ??????????ƚ?u?a?U?Q?P?V?\?hƁƎƧ????????????????ùîìèìóù??????????????? ???????????????????????????????????H?M?U?Y?Y?U?H?D?D?D?H?H?H?H?H?H?H?H?H?H?/?<?F?H?L?H?A?<?/?'?&?-?/?/?/?/?/?/?/?/?H?T?a?o?r?j?Z?G??	????????????"?/?=?H???????????????????y?T?G?;?2?0?4?@?S?m???-?:?A?F?I?J?F?:?0?-?+?!?-?-?-?-?-?-?-?-ùù????????ùðìàÓÏÊÓØàìøùù?-?:?E?F?S?U?S?R?G?F?:?2?-?,?'?'?-?-?-?-?3?3?7?@?L?O?Y?e?Y?L?@?3?3?3?3?3?3?3?3?3?????üҼռѼʼ???????????{?~?????????????????"?.?:?B?E?;??	?????????????????????????????????? ?????????????????????????????????????????????ҿѿݿ?????????	???????ݿѿ˿Ŀ??Ŀѻ?????????????????ֻܻϻλǻлջ??????????????????????????????????????????)?-?.?)????????????????????z?{?z?z?t?n?f?a?[?]?a?m?n?z?z?z?z?z?z?z?a?m?z???z?z?m?g?a?T?K?M?T?_?a?a?a?a?a?a????????????????????????????????????????????????ƾƾ???????f?Z?M???@?M?s???????#?0???D?B?=?3?#????????????????????B?h?tĒĥĠĔ?u?p?m?h?[?6?+??%?5?8?8?Bìù??????????????ùàÎÇ?z?yÀÌÓàì?A?M?Z?f?h?g?f?Z?M?A?@???A?A?A?A?A?A?A?A?f?r?r?r?n?j?f?]?Z?W?M?I?J?M?P?W?Z?c?f?f?r?~???????????ƺ˺????????~?v?g?a?b?l?r?A?N?Z?[?g?q?g?g?g?Z?N?D?A?;?A?A?A?A?A?A?T?a?h?m?q?x?z?}???z?m?a?^?X?P?O?J?O?T???Ľнݽ????????ݽĽ??????|?????????????????????$?4?:?8?"?	?????????????????????
???#?%?%?#? ??
???????????
?
?
?
?????????????????????????????????????????(?5?>?=?5?-?(????? ?(?(?(?(?(?(?(?(????????????????????????r?f?`?c?f?r???D?D?D?D?D?D?D?D?D?D?D?D?D?D?D{DzDvD{DD??????????????????????:?F?S?_?l?x???x?w?l?e?_?S?H?F?:?2?.?:?:??*?6?@?C?O?R?O?O?C?6?*?"????????????ûлջлû??????????????????????????{ŇŔŖşŝŔōŇ?{?n?b?U?O?V?_?h?n?y?{ Y , ? \ N J A m 0 - J 7 5 8 O 5 ( , d ; x C ] d ] B E V 9 6 D ; B    - ? j $ [ 9 P  0 j 4 j k j c # _ X # ( \ C 6 M C  N  t  S  ?  ~  5  ?  ?  6  ]    |  2  g  e  9  V  ?  ?  x  m  s  D  I  ?  ?  ?  U  ?  ?  ?  ?  3  t  3  ?  ?  ?  %  ?  ?  ?  J  ?  O  >  ?  ?  ?  ?  ?  ?  ?  ?  k  ?  B  ?  U  ?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  B?  v  q  l  g  b  ]  X  R  M  H  ?  1  #       ?   ?   ?   ?   ?  y  v  s  h  ]  O  ?  -      ?  ?  ?  ?  x  U  0    ?  ?       ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?   ?      #  +  2  6  :  >  A  A  B  B  A  ?  <  :  .         ?  ?  ?  ?  ?  l  X  D  )    ?  ?  ?  ?  `    ?  `     ?  ?  ?  ?                ?  ?  ?  ?  ?  E  ?  e  ?   ?  Y  f  c  O  /  	  ?  ?  ?  G  	  ?  ?  >  ?  ?  A  ?  L  n  ?  ?  ?  ?  ?  ?  l  T  >  (    ?  ?  ?  ?  R  ?  8   ?   @  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  r  \  J  9    ?  ?  ?  D    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ^  :    	  ?  r  ?  ?    ?  ?  n  O  ,  	  ?  ?  ?  f  8    ?  ?  [    ?    3   ?  /  )  #              ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  P  C  5  &    ?  ?  ?  ?  ?  ?  \  1  ?  ?  P  ?  >  ?  ?  ;  ;  :  ;  <  =  >  <  ;  5  .  $      1  @  +    ?  ?  ?  ?  5  ?  ?  ?  ?      *  7  A  C  C  E  C  G  U  k  ?  ?  ?  ?  y  q  i  _  U  K  @  5  (    ?  ?  ?  |  Q  %  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  m  R  4    ?  ?  ?  a  -  ?  ?  ?  ?  ?  ?  ?  x  x  ?  ?  ?  ?  P    ?  u  ?  *  j  ?  ?  \  O  I  H  J  D  =  5  +      ?  ?  ?  ?  ?  )  ?  L  ?  ?    M  U  ^  h  q  t  u  w  z  |  |  {  w  v  z  c  H  *  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  s  a  K  /    ?  ?  ?  ~  \  :      .  ?       ?  ?  ?  d  0    ?  ?  ?  ?  Y  *  ?  ?  ?  /      ?  ?  ?  ?  ?  ?  ?  ?  `    ?  b  ?  i  ?    ?  ?  b  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  g  W  >    ?  ?  ?  a  '  ?    y  ?  ?      #  (      ?  ?  A  ?  y    ?  n  e  K  y  ?  ?  ?  ?  ?  ?  ?  z  j  Z  H  B  6      ?  ?  ?  ?  b  ]  X  S  N  H  >  5  +  !    	  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  i  I  (        ?  ?  ;    ?    R  l  o  k  g  j  f  W  @  $     ?  ?  k  0  ?  ?    ?  ?  ?  z  r  v  y  {  |  |  |  {  z  u  q  l  g  a  Y  Q  H    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  e  V  F  8  -                  ?  ?  ?  ?  ?  ?  ?  t  J    ?  ?  r  L        ?  
        ?  ?  ?  ?  ?  p  4  ?  ?  2  ~  ?  ?        
  ?  ?  ?  ?  ?  \  4    ?  ?  Q    ?  ?  G  ?  &  T  j  x  {  v  h  Z  I  /    ?  ?  ?  n  1  ?  ?  =  ?  ?  ?     +  S  r  ?  ?  ?  	  	d  	?  	?  
<  
?    _  9  n  Y  Q  H  @  ;  6  1  2  7  ;  2      ?  ?  ?  ?  t  i  ^  |  r  g  ]  P  D  7  (    
  ?  ?  ?  ?  ?  ?  ?  ?  q  [  E  B  B  3       ?  ?  ?  ?  ?  w  G    ?  w  ;    ?  ?  
?  G  ~  ?  ?  ?  ?  ?  ?  ?  J    
?  
G  	?  ?    ?  ?  F  ?  ?  ?  ?  ?  ?  s  m  ~  ?    g  :    ?  {  #  ~  ?  ?    	  	a  	?  	?  	?  	?  	?  
  
  	?  	?  	?  	?  	{  	  T  W    	    m  Z  A  &    ?  ?  ?  ?  t  X  ?  *     #  T  ?  ?  -  )  G  :  %  v  P  )    ?  ?  ?  k  =    ?    ?  \  ?  5  R  ?  ?  ?  ?  t  P  6    ?  ?  E  ?  ?  2  ?  	  i  ?  1  ?  ?  ?  ?  ?  ?  ?  z  n  `  R  D  8  +         %  F  g  ?    ?  ?  ?  ?  ?  ?  ?  ?  ^  .  ?  ?  v  7  ?       ?    ?  ?  ?  x  \  ?  "    ?  ?  ?  ?  g  C    ?  ?  ?  Y  ?  ?  ?  ?  ?  ?  U  $  ?  ?  ?  9  ?  ?  ?  ?  z  ?  {  ?  ?  ?  ?  ?  ?  ?  ?  ?  m  Y  A  $    ?  ?  ?  ?  n  X  A        ?  ?  ?  ?  ?  +  =  4  $    ?  ?  ?  ?  ?  ?  n  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  j  ^  Q  E  8  ,  ?  ?    N  t  ?  ?  ?  ?  ?  #  (    ?    6  ?  
$  %  F  &  i  ?    ?  ?  ?  ?  h    ?    R  ]  L  -  ?  y  ?    n  z  |  e  O  8       ?  ?  ?  m  8  	  ?  ?  ?  ?  m  O  ?  ?  r  X  ?    ?  ?  ?  U    ?  ?  u  E    ?  ?  1  ?  }  >    ?  ?  ?  ?  c  *  ?  ?  :  ?  {    ?  7  ?  J  q  ?  ?  X     ?  ?  z  A    ?  ?  T    ?  ?  H  ?  ?  k  !    ?  ?    ?  |  %  
?  
k  
  	?  	"  ?  S  ?    J  o  ?  ?