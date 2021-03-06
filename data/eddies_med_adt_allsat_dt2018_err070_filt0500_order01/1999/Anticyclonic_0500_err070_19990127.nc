CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ?ě??S??      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N??   max       P?8?      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ????   max       >?      ?  ?   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>Tz?G?   max       @E??z?H     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??(?\    max       @vrz?G?     	`  )?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @.         max       @O            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?        max       @??`          ?  3?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??/   max       >/?      ?  4?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??)   max       B,??      ?  5?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A??/   max       B,      ?  6?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?0?   max       C?Z?      ?  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?0??   max       C?I      ?  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ?      ?  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      ?  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      ?  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N??   max       P?{?      ?  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??&???   max       ??5?Xy=?      ?  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ????   max       >?      ?  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??
=p?   max       @E??z?H     	`  >?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??(?\    max       @vrz?G?     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @         max       @O            x  Q?   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?        max       @???          ?  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F   max         F      ?  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???vȴ9X   max       ??4?J?     ?  T      ?                  5         )      Y         	      4   p   p      8            G      	            /   )         @               	      ,      f   -         
      5               
   D         NN8Pu?{Nw;O??OZ?sN)20N#+?P??N+?'N?-OP??N[D?P?N?j;N8.WN?H?N??O?1?P?^P?8?N??P(!?O??OO?4N??]PO?N?@?NɨpN??OVߴPJv`P4ZO??:O???O?;>O]J?N???N?5?N$;?N?RgOk6P*h%N??uP)??OY??N?Y.N?k6N???N?W?O?>?N%??N1??NP??Nyh?Ng|1O??NU?*Na??Ns?N??????/???????ͼe`B?T???#?
??o??o??o:?o;?`B<#?
<#?
<T??<T??<T??<e`B<e`B<u<?o<??
<?1<?1<?j<ě?<???<?/<??h<??h<??h<???<???<???=o=o=\)=??=49X=49X=8Q?=@?=D??=D??=H?9=H?9=L??=P?`=aG?=q??=q??=u=???=??
=??T=? ?=ě?=ě?=ȴ9>?????????????????????!!*BN[gx???tgNB5/!sst???????ztssssssss
#*/265/#"
jkp?????????????zrnj????????????????????*)00<GB<90**********?????	";@;8<<7/"
??)!)56>95)))))))))))rpont????????trrrrrr?#/<BHIJHF</#
??????????????????????????)B[ce]NB5)???pt???????????~xtpppp@BNU[ab[QNGB@@@@@@@@KHO[hqtuutmh[YQOKKKKaabcnwxunbaaaaaaaaaa??????? 
??????????????
*<ISSIC0????????????-64==5)??????????????????????ns?????????????????n?????
 "##&#
???PNOQQUXanz}?~ynaZUP../6<EHRRRH<9/......Z\??????????????wjbZ"/;HPTVXUQH;/"????????????????????:8<HKUXXWUSOKHFA>>>:post????ztpppppppppp???????????????????????????????????????????????
'/;4/#???  #&(/<Uac^[VPH</$ ???????????????????????????
#&#!
????;9<BHKTamuz}}zsmaTH;???????????????????????????
	????

???????????????????

????????????)3@BA8)????#/<><4/-#??????????????????
		).5:@A75)$????????????????????gghijklpt???????zthg????(????
!#$#
#/<CFHJHD>/#GDFHUX\YUHGGGGGGGGGG????????????????????QTWamxz|zmaTQQQQQQQQ????????????????????C<HTaabaTHCCCCCCCCCC?????????????????????((??????????_UUamsupma__________?????????????????????=?I?T?N?I?=?0?-?0?5?=?=?=?=?=?=?=?=?=?=??)?B?O?c?o?v?t?h?[???????????????????zÇÉÈÇ?|?z?q?n?j?n?u?z?z?z?z?z?z?z?z???(?5?A?@?5?1?(? ?????????? ??
?????.?5?8?5?(???????????????????#?/?<?<?<?:?0?/?.?%?#??#?#?#?#?#?#?#?#?ݽ???????????߽ݽ׽ݽݽݽݽݽݽݽݽݽ??"?/?H?P?m?m?q?n?a?Y?H?;?"?	???????	??"¦©§¦?r?????????????????r?i?f?o?r?r?r?r?r?rD?D?D?D?D?EEEED?D?D?D?D?D?D?D?D?D?Dӿ"?.?:?;???;?.?#?"?!???"?"?"?"?"?"?"?"????????????????ƳƎ?W?O?Y?{ƋƳ???T?Z?a?a?a?\?T?K?H?;?3?1?8?;?H?M?T?T?T?T???????????????????????????????????????Ѿ׾??????????ؾ׾ʾɾ????????ʾѾ׾׾׾׾?(?1?4?<?4?(????????????????ûɻû??????????x?_?F?=?D?_?l?x????????????!?-?5?6?-?????????غɺ??????????I?[?h?f?j?zŇŪŪŝŇ?n?<???????????E?I???????????????????????????????????????????#?*?6?B?4??齫?????y????????̽???????????????žɾ??????????????????????????????????Ⱦʾо˾???????????x?y?????????????
?????????????????????????????s???????????????Z?A?5?'???"?/?A?N?Z?s?????
??#?*?0?1?4?0?)?#??
?????????????M?Z?f?l?h?m?s?~?s?f?_?Z?M?L?F?J?M?M?M?ME?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E滪???ûŻλû??????????????????????????????????????ĽȽнĽ??????????y?m?q?s?~???????????????????????g?B?.?'?'?5?A?g???????"?/?;?E?U?K?D?;?5?"??	???????????????????????????????????????????????y?x??????????(?A?M?f???????s?f?A?#???????)?5?N?]?d?d?b?[?B?,?)? ?????????????(?.?5?A?F?E?A?<?5?(?????????àëìù??????þý??úùìàÖØØ×ßà?_?l?x?????????????????????x?l?h?_?S?_?_?????Ⱦɾ????????????????????????????????׾????????	??	?	???????׾ϾʾȾʾҾ׾?¦²¿????????????????????¿¾²¨¦¤¦?????ݿ??ѿſ??????y?`?T?;?2?3?F?S?m?????нݽ??????????????ݽӽнǽ˽ннннн?Óàù????????à?z?a?W?]?Z?G?G?U?n?{ÈÓ???
??#?H?R?H?@?<?5?/?#??
???????????????????????????ܹܹܹܹ????????????-?:?@?F?S?_?l?x???x?l?b?_?S?F?@?:?1?+?-???!?-?0?-?!???
????????????????y?????????????????~?y?t?m?h?m?u?y?y?y?y?	??"?.?G?`?h?a?X?G?;?.??	???????????	?B?N?[?_?^?[?N?B?7???B?B?B?B?B?B?B?B?B?B???)?0?5?)??????????????āĄčĔĒčċā?|?x?z?āāāāāāāā???????????????????????}???????????????????????????????????????????????????????̼r??????????????????????????|?p?c?]?f?r????????????????????????????????????????ŠŭŹ????ŹŭŠŞşŠŠŠŠŠŠŠŠŠŠDoD{D?D?D?D?D?D{DwDoDkDoDoDoDoDoDoDoDoDo J ! ? D M Y [ 6 A = I 4 6 K s + 4 ^ ( M : d 6 , % 3 ! [ w [ 5 Y K 9 /  8 W [ 2 D h N P I > 3 V r G T ; : _ c = a | $ 8  *  ?  .  R  ?  j  K  i  5  ?  ?  s  ?  ?  m  ?  4  ?  ?  3  ?  v  7  ?  ?  ?     ?  :  )  ?  ?  ?    =  ?  ?  B    A    ?  7  ?  $  ?  ?  -  ?  ?  b  6  R  ?  ?  r  ?  ?  e  s??/>	7L???
?#?
;D???o???
=T??;?o<49X=,1<T??=???<?o<?t?<?1<???=?+>   >%<?9X=???=#?
=D??=?P=??=aG?=t?=0 ?=?w=8Q?=??-=?t?=}??=aG?=\=?%=P?`=ix?=<j=Y?=?o=?j=u>??=??=}??=??-=?o=?\)=?`B=?+=?9X=?9X=?-=ě?>&?y=?/=??>/?B B?<B
=?Bj?B??B{?B%?OA??)B??B? B??B>B?7B
??BJ?Bq?B()?B"??B$p|B3LB??B ??B?dB?B??B
?/A?>Ba?BBpB+? B?BQVB?B6B?dA?#?B!6B?gB$t3B?-B?xB?BBѴB??BI?B?-BZ{B@?B:BD?B??Bh?A??KB,??A?DSB?B,?A??{BXB?BcB
@}BA?B??BDMB&?A??/B;?B?YB??B
?|BA?B
??B??B?%B(??B"??B$BjBNoB?FB ϺB??B??B??B
ɊA??B\?B??BK{B+?tB?B??B?mB6B-A?zTB!??BއB$?B??B?CB<?B??B??B?+B?BF?B=LBEbB?UB??BALA??B,A???B?IBU?A??zB=?B
߭A?t`AȍCA?`?A???A??A-6oA?R?A??R@??}C?2nA`>VB2oA???A??
AR?&A6Ё@???@G?A?5?Arm?A+Q:AJ??AJ?$A???A???A??}A@_?C?Z?@?V?A XA?qA?i?A??!A=?BA??OA???A̿?@?I?AM??AV8?A?!?An?cA+n]A?|A?u??0?@?ş@bO?Ao?A_,A?hbA??PAݣ?@?;B??@??@??rA??)C??\B
??AԂA?{?A?p?A?OA?~?A-?bA??#A?wI@???C?7?A`??B??A?\A???AQ??A6?K@? ?@DI?A??MAr?cA)??AJ??AK A?|0A??0A??JAA1?C?I@???A??A??`A?oA???A=?\A?TdA?u?A?|?@?wYAN
AUaA?Y?Ak?A*?UAʂ%A?u??0??@?
0@[EAoo?A^6A?~?AՃA݀&@??MB?@??@? )A??dC??      ?                  6         *      Z         	      5   p   p      9            G      
            /   *          @               	      -      g   .         
      6         	         D      	         1                  '               ?               %   )   C      3            )                  3   )      &   !                     /      /                                                   %                                 7                     5                  !                  3                                 /      #                                             NN8P
y?Nw;O??O?N)20N#+?O]}?N+?'N?-O-?N[D?P?{?N?j;N8.WN?H?N??O??O??P?a3N??OIPsO??O
?^N?_O??Ov??N?@?NɨpN??OVߴPJv`O?(HOY??O??O?6?O]J?N???N?5?N$;?N?RgOk6P*h%N??uO??1OY??N?Y.N??N???N?W?OxSN%??N1??NP??Nyh?Ng|1O??NU?*Na??Ns?N    c  F  ?  B  T  J  &  ?  ?  	(  ?  ?  ?  E  ?  ?  |  M  
,    D  ?  ?  ?  ?  ?  L  F  ?  /  ?  ?  ?  C  ?  ?  ?  ?  v      ?  1  ?  
?      ?  N  	?  ?  ?  |    D  w  J  <  
m????<??ͼ??????ͼ49X?T???#?
<?9X??o??o;ě?;?`B<?t?<#?
<T??<T??<T??<?9X=??=?P<?o=@?<?1<?`B<???=?P<???<?/<??h<??h<??h<???=49X=t?=?w=T??=\)=??=49X=49X=8Q?=@?=D??=D??=?t?=H?9=L??=Y?=aG?=q??=??=u=???=??
=??T=? ?=ě?=ě?=ȴ9>?????????????????????-+.4>N[gty???tg[NB5-sst???????ztssssssss
#*/265/#"
motz????????????zynm????????????????????*)00<GB<90**********???	"(-01."	?)!)56>95)))))))))))rpont????????trrrrrr
#(/<=EFB<7/#
?????????????????????????BXaa[HB5)????pt???????????~xtpppp@BNU[ab[QNGB@@@@@@@@KHO[hqtuutmh[YQOKKKKaabcnwxunbaaaaaaaaaa??????????????????????	
##)*%#
???????$--160) ????????????????????????????????????????????????
 "##&#
???SSUXanwz{|{zxpnfaWUS///8<BHQPMH<:///////c`eo?????????????sjc"/;HNTUWTPH;/"????????????????????:8<HKUXXWUSOKHFA>>>:post????ztpppppppppp???????????????????????????????????????????????
#,)#
???? "#&*/<HUY[XSMH</)# ???????????????????????????

?????;9<BHKTamuz}}zsmaTH;???????????????????????????
	????

???????????????????

????????????)3@BA8)????#/<><4/-#????????????????????
		).5:@A75)$????????????????????hjklmqt}????????xthh????(????
!#$#
#/<@DFHFA</#GDFHUX\YUHGGGGGGGGGG????????????????????QTWamxz|zmaTQQQQQQQQ????????????????????C<HTaabaTHCCCCCCCCCC?????????????????????((??????????_UUamsupma__________?????????????????????=?I?T?N?I?=?0?-?0?5?=?=?=?=?=?=?=?=?=?=???6?B?O?]?a?]?P?B?)?????????????????zÇÉÈÇ?|?z?q?n?j?n?u?z?z?z?z?z?z?z?z???(?5?A?@?5?1?(? ?????????? ??
?????!?(?*?0?2?(????????????????#?/?<?<?<?:?0?/?.?%?#??#?#?#?#?#?#?#?#?ݽ???????????߽ݽ׽ݽݽݽݽݽݽݽݽݽ??/?;?H?T?a?a?b?a?]?T?;?/?"??????"?/¦©§¦?r?????????????????r?i?f?o?r?r?r?r?r?rD?D?D?D?D?EEE	EED?D?D?D?D?D?D?D?D?Dӿ"?.?:?;???;?.?#?"?!???"?"?"?"?"?"?"?"?????????
???????ƷƎ?u?e?jƁƋƓƳ???T?Z?a?a?a?\?T?K?H?;?3?1?8?;?H?M?T?T?T?T???????????????????????????????????????Ѿ׾??????????ؾ׾ʾɾ????????ʾѾ׾׾׾׾?(?1?4?<?4?(??????????????????????ûŻ??????????x?l?X?V?_?l?x?????ֺ???????????????????ֺԺɺɺƺɺ̺??<?U?nŇřŜŒŇ?n?U?#????????????????<?????????????????????????????????????????ݽ????????????ݽнĽ??????????Ľݾ????????????žɾ????????????????????????????????ƾž???????????????~??????????????????????????????????????????????g?s?????????????????s?Z?A?7?8?3?A?N?Z?g?????
??#?(?/?0?3?0?'?#??
?????????????M?Z?f?l?h?m?s?~?s?f?_?Z?M?L?F?J?M?M?M?ME?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E?E滪???ûŻλû??????????????????????????????????????ĽȽнĽ??????????y?m?q?s?~???????????????????????g?B?.?'?'?5?A?g?????	??$?;???B?@?7?/?"??	???????????????	???????????????????????????????}?|???????M?Z?f?w?}?}?z?s?f?Z?A?9?4?,?%?(?-?4?<?M??)?5?B?N?\?Z?S?N?B?5?)????????????(?.?5?A?F?E?A?<?5?(?????????àëìù??????þý??úùìàÖØØ×ßà?_?l?x?????????????????????x?l?h?_?S?_?_?????Ⱦɾ????????????????????????????????׾????????	??	?	???????׾ϾʾȾʾҾ׾?¦²¿????????????????????¿¾²¨¦¤¦?????ݿ??ѿſ??????y?`?T?;?2?3?F?S?m?????нݽ??????????????ݽӽнǽ˽ннннн?Óàìù????????úàÓ?z?n?e?[?V?Z?zÇÓ???
??#?H?R?H?@?<?5?/?#??
???????????????????????????ܹܹܹܹ????????????:?F?S?_?l?x??x?v?l?`?_?\?S?F?A?:?3?:?:???!?-?0?-?!???
????????????????y?????????????????~?y?t?m?h?m?u?y?y?y?y??"?.?;?G?X?Y?Q?G?;?.??	???????????	??B?N?[?_?^?[?N?B?7???B?B?B?B?B?B?B?B?B?B???)?0?5?)??????????????āĄčĔĒčċā?|?x?z?āāāāāāāā???????????????????????}???????????????????????????????????????????????????????̼r??????????????????????????|?p?c?]?f?r????????????????????????????????????????ŠŭŹ????ŹŭŠŞşŠŠŠŠŠŠŠŠŠŠDoD{D?D?D?D?D?D{DwDoDkDoDoDoDoDoDoDoDoDo J  ? D K Y [ 5 A = J 4 / K s + 4 S  < : e 6 4      [ w [ 5 Y ? 9   8 W [ 2 D h N P < > 3 I r G I ; : _ c = a | $ 8  *  b  .  R  c  j  K  ?  5  ?  ?  s  ?  ?  m  ?  4  @  7  V  ?  ?  7  -  ?  ?  ?  ?  :  )  ?  ?  S  ?  ,    ?  B    A    ?  7  ?    ?  ?    ?  ?  ?  6  R  ?  ?  r  ?  ?  e  s  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F        #  '  +  .  0  3  5  9  <  @  D  H  L  O  S  W  [  ?  p    ?    P  b  S    ?  Z  ?    2  L  
P  	>  ?  y    F  7  (    ?  ?  ?  ?  ^  ;    ?  ?  ?  ~  W  /     ?   ?  ?  ?  ?  ?  {  v  u  r  m  d  Y  I  6  !  	  ?  ?  ?  ?  ?  !  2  <  A  <  0  !    ?  ?        ?  ?  ?  ?  ?  ?    T  D  4  $    ?  ?  ?  ?  ?  ?  v  `  K  7  "  !  &  +  0  J  >  2  %      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  e  W  J  ?  H  ?  "  x  ?  ?    $  #  ?  ?  Z  ?  ~  ?  v  ?  ?    ?  }  v  o  g  W  F  6  )  "      
    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  w  p  g  Z  F  2      ?  ?  ?  ?  	  	#  	(  	%  	  	  ?  ?  ?  ]    ?  ;  ?    _  ?    !  *  ?  ?  ?  ?    s  g  Y  L  >  >  L  [  g  n  v  ~  ?  ?  ?  x  ?  ?  ?  ?  w  e  V  E  ,    ?  ?  J  ?  b  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  n  ^  N  >  .      E  =  5  -  '  +  /  3  5  3  1  /  &      ?  ?  ?  ?  |  ?  ?  ?  ?  ?  ?  ?  ?  y  g  T  >  (    ?  ?  ?  ?  ?  ?  ?  ?  w  m  `  R  D  5  %      ?  ?  ?  ?  ?  ?  b  >      J  t  {  t  `  A      ?  ?  ?  ?  ^    ?  2  ?  5  ?  }  *  ?  	E  	?  
9  
?  
?    >  M  9    
?  
D  	?  ?  m  *  ?  	
  	?  
  
)  
*  
  	?  	?  	?  	C  ?  ?  9  ?  D  H  ?  ?  ?  ?        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  ?  ?  ?  ?         0  B  <    ?  ?  ?  O    ?  ?    ?  ?  ?  ?  ?  ?  ?  ?    X  &  ?  ?    C  ?  ?  T    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  P  '  ?  ?  ?  r  D    	  ?  ?  ?  ?  ?  ?  ?  ?  d  4  ?  ?  ?  ]  *  ?  ?  ?  ?  ?  ?  M  ?  ?  ?  ?  ?  ?  ?  ?  ?  R    ?  0  ?    k  ?  2  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  [  )  ?  ?  )    ?  ?   ?  L  E  >  4  +      ?  ?  ?  ?  t  S  /    ?  ?  ?  i  O  F    ?  ?  ?  ?  p  ;    ?  ?  f  *  ?  ?  a    ?  ?  8  ?  ?  ?  ?  ?  ?                   ?  ,  ?  5  ?  8  /  %      
    ?  ?  ?  ?  ?  ?  ?  ?  ?  h    ?  o  *  ?  ?  ?  ?  x  <  ?  ?  ?  q  -  ?  m    ?  ?  9  ?  ?  M  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  P  ?  ?  ?  ?  |    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  E    ?  n    ?  ?  ?  ?  U  ?  ?    !  9  @  C  @  8  '    ?  ?    =  ?  q  ?  n  	  w  ?  2  ~  ?  ?  ?  ?  ?  ?  Z    ?  O  ?    f  ?  ?  ?  ?  j  j  a  L  1    ?  ?  ?  s  A    ?  v    ?  /  ?     ?  ?  n  V  <        ?  ?  ?  r  2  ?  ?  ?  Z    ?  ?  ?  ?  q  a  P  ;  #    ?  ?  ?  ?  a  $  ?  ?  x  [  h  ?  v  n  g  _  X  P  I  A  :  2  .  -  ,  ,  +  *  *  )  (  '      ?  ?  ?  ?  ?  ?  n  U  <  "    ?  ?  ?  E     ?   ?      ?  ?  ?  ?  ?  w  T  ,  ?  ?  x  5  
  ?  ?  ?  ?  P  ?  ?  ?  ?  ?  ?  ?  z  i  N  /  ?  ?  ?  F  ?  p  ?  v  ?  1    	  ?  ?  ?  ?  ?  u  _  G  .  "    ?  ?  ?  x  I    ?  l  ?  ?  ?  ?  ?  ?  ?  Q  ?  ?    X  
?  	?  w    i  ?  
?  
?  
}  
l  
a  
T  
<  
.  
  	?  	?  	  ?  .  ?  ?  <  ?  ?  k        ?  ?  ?  ?  ?  ?  ?  ?  c  3  ?  ?  ~  8  ?  ?  U  ?  ?    ?  ?  ?  ?  R    ?  ?  r  ;    ?  ?  ?  P  ?  }  ?  l  V  >  %  
  ?  ?  ?  ?  x  Z  <  "    ?  ?  R    ?  N  7  !    ?  ?  ?  ?  ?  ?  m  W  @  +      ?  ?  ?  ?  	?  	?  	?  	?  	?  	?  	?  	?  	l  	3  ?  ?  `    ?  ?  ?  ?  ?  g  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  o  f  \  R  I  A  :  2  ?  ?  ?  ?  ?  ?  ?  ?  ~  q  e  [  R  G  X  n  ?  r  X  9  |  u  n  c  W  J  >  2  H  d  n  h  a  Y  P  G  =  3  (      ?  ?  ?  ?  ?  ?  f  F  &    ?  ?  ?  ?  ?  r  q  p  o  D  2      ?  ?  ?  ?  k  C    ?  ?  ?  R  !  ?  ?  ?  P  w  j  \  H  5    ?  ?  ?  :  
?  
}  	?  	q  ?  B  X  *  ?  ?  J  -    ?  ?  ?  ?  n  F    ?  ?  ?  _  .  ?  ?  ?    ?  <  !    ?  ?  ?  ?  z  ]  @  "    ?  ?  ?  ?  m  H    ?  
m  
9  
  	?  	?  	@  ?  ?  C  ?  ?    ?     f  ?  N  ?  *  ?