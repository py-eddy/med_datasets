CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�t�j~��     H  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�    max       P` �     H  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��{   max       <u     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>c�
=p�   max       @F�\(�     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v��
=p�     �  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @N�           �  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�=@         H  ;�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       <49X     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��d   max       B4�8     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�p   max       B4�n     H  ?�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�~�   max       C��     H  @�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��r   max       C��     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          M     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7     H  D�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�    max       PG��     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�y=�b�   max       ?��u%F     H  H�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��{   max       <e`B     H  I�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>c�
=p�   max       @F�\(�     �  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v�\(�     �  W�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @N�           �  d�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�9        max       @�~�         H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     H  f�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��7��3�     �  g�         
                     (            
   L                                           
         (      !   -               	      $            %   
   	            	                     )   $         !         
                     +                     +      N��5N��O
y�NGd\N���Nc\WN�O���N^\P�LOa�hO��N��hO+3�PIH�Nڇ<N���O�v�O	��O�j&N�ضO-�O���O<U�O��O\YN��O(��O`�N���O?l.P/�O�BOR��P` �O��N�šN�V�O{n@N��/N:HsO֪�O'>xN�ON�6�O��Oa�N��IP T�M� O�g�OY�N�OO�k_N�^�N��N��/O�O��EOވ9N��qNx�O��KNn�xN�<�N�׌N�:N�	^O���N�G�N<�CO�O� N�W�N}��N�R�O�O��hN��O��N��N`�<u<e`B<T��<T��;�`B;�`B;�`B;ě�;�o:�o:�o:�o%   ��o�o�D����o���
�ě��ě���`B�49X�D���T���e`B�e`B�e`B�e`B�u��C���t����㼛�㼛�㼣�
���
��9X��9X��j�ě����ͼ�/��/��`B��h��h�����o�o�o�+�\)�\)�\)�t��t��t���P������w��w�',1�8Q�@��@��H�9�L�ͽL�ͽL�ͽT���]/�aG��aG��}󶽃o��O߽��-���-��{
 #+019<A<0#
��������������������=HOUahnz����znaUJH<=��������������������������������������������������������������������������������y������������|{zqqry���������������������������������������
#,...,#
����6<>HIUacgga_ZUH><166������	�����������

�����)2BN[h�����gNFB5)!)rz}��������������{zr������������������������
"..'
������RTUX]ampnqssrqmaWTRR�
)0@BPROB)���9BFOX[d[[ONB<7999999���
������������������������������rt����������xtpkhimr#/<HUXUOH<#
#'/<HNQMPJH<:/'# UU`abnnnhaVUUUUUUUUU$*46BDNORSQROC6*����� 		����������������

�����������������������������;>HTamvz��}tmaTHB;; )6BO[`hntti[OEB6-) ������������������������#OUn{}xf<0
���������������������������
 
��������������������������������������hht������tha]\hhhhhhW[bhioih[[UUWWWWWWWW������������������������
#%)#
����������

������������������������������������$&#
����������������������������������������������lt��������������}tml��������������������^ilz�����������za[Y^aaekmz��������zmkeaalmrz��������~zzmkjjl�������������������������������������������"!��������������������������-0<IKQTUVUUQI<60,))-&+6AO[hqsppnih[OB6+&���������������������(*21*��������

		
���������� &0<IUYbcb^UNI<. 	
#'/;/*#!
						uz{��������������zuuNOY[dhhplh][YONMNNNNagt���������thg]aaaa��������������������������)7@B6)$��������������������������������������������������������������#/<HRn}����zrUH</(!#������������������������������������������������������������jt}������������trmjj )B[gg[VNB5) !)+*) ������������)-44/*)���������������������������������������¼ʼϼ˼ǼǼ����������U�K�J�M�Q�U�`�a�n�o�n�m�g�a�U�U�U�U�U�U�4�.�(�#�$�(�-�>�A�G�M�P�Q�N�P�S�M�L�A�4���������������ʾԾվϾʾ����������������������������������	���	�������������˽������������(�)�(��������������������������!�%�#�!���������������������������������"�6�;�?�B�6�*���������h�g�f�h�uƁƎƁ�{�u�h�h�h�h�h�h�h�h�h�h������ƮƤƭ������0�1�'�"����������N�A�5�)�(�(�*�2�5�A�N�Z�g�o�r�q�m�g�Z�N�(� ��������(�2�5�A�L�H�A�@�5�(�(�z�p�n�a�U�R�O�U�\�a�c�n�y�zÇËÇÂ�{�z�A�:�4�(��(�4�A�E�M�Z�f�s�w�z�s�f�Z�M�A���������������ҿ���
������ԿѿĿ���������ü������������������������������ùöìàßÕÖßàìù����������ÿýùùàÓÌËÑ×àìù����������������ùìà�t�i�h�[�O�B�7�B�O�[�h�tāčĔĖčĉā�t�Z�M�B�A�K�Z�s���������������������f�Z�U�Q�U�Y�a�e�n�z�z�}�z�s�n�a�U�U�U�U�U�U������������������������������޾A�;�A�M�R�I�A�A�_��������������s�]�M�A�׾Ծо̾;׾��	����"�(�"�	������²®²³����������������������������¿²���������������������������������������������������������������������������������"��	�����������	��"�$�.�4�8�.�"�	����������	���/�:�;�H�M�H�=�/�"��	�������������������������������������������������������)�6�@�I�O�T�B�6�)�Ɓ�u�g�a�c�i�u�~ƚƳ���������������ƧƁ�m�g�^�_�X�[�`�h�m�x�z�������������y�t�m�a�_�U�R�H�D�<�5�4�<�H�U�a�n�u�|À�n�e�a�������������g�`�R�Q�����������������m�`�S�J�D�G�T�`�m�y�����������������y�m�лȻŻû»»ûлܻ�����������ܻл���������������������������׾ʾþ����ʾ׾���������	���������������������������������������������L�B�L�T�Y�e�r�r�r�r�e�Y�L�L�L�L�L�L�L�L����Ŀ����������0�I�b�a�d�U�<�#��������0�*�$�����
���$�.�=�A�I�K�N�I�=�0���������~������������������������������ŠŖŔŋŔŕŠŤŭŹźźźŹůŭŠŠŠŠ�������������������ɺں�����������ɺ��{�u�q�o�d�\�b�g�o�{ǈǎǔǙǘǔǍǋǈ�{¦£¦²µ¿¿¿¿²¦¦¦¦¦¦Ŕŉŉŀ�|ŗţŞŠ��������������ŹťŠŔùðìàÕàìù������ûùùùùùùùù�m�a�T�F�>�;�D�a�m�z����������������z�m�"�!��	�������	���"�/�;�<�F�A�;�/�$�"������üù÷íù���������������������������� ����5�B�N�[�a�`�Y�J�B�5�)������������~�x�v�x����������������������������������������������������������������čĈā�t�h�b�^�h�tāčĚĝĦĲĦĚďčč�� ��������$�(�4�4�A�L�J�A�4�(�����ܹҹù������ùϹܹ����	���������輽�������v�t�������ʼּܼ�������ּʼ��������������!�.�1�0�.�.�!��������Ľ½����������Ľнݽ�ݽݽнĽĽĽĽĽĽ����w�s�w�����������Ľνн׽нĽ�������ECE=E:EAECELEPEZE\E^E\EWEPEFECECECECECEC�������������������������ĿſĿ������������������������ûлӻܻ�ܻһлû��������������������	��"�"�"�"����	���������	�����������	�������	�	�	�	�	�	�н������l�b�j�y���������Ľǽսݽ���ݽлS�M�F�B�C�F�S�_�l�s�x��������x�l�_�S�S������������!�)�,�!����������?�4�1�4�5�@�M�Q�Y�a�f�r�x�~�r�l�f�Y�M�?F
F	FFFF8F=FIFcFoF�F�F|FuFnFaFGF1FF
��������������!�������������������������'�2�3�?�3�'����������~�t�t�~���������������Ǻ��������������~����ĿľĿ����������������������������ÓÊ�z�n�c�^�K�E�H�a�n�zÈÐÕÝÝåâÓ�����������������������������������������Y�M�@�9�5�:�@�M�f�r��������������r�f�Y�������������'�(�3�-�'�������/�,�)�/�<�H�L�Q�H�<�/�/�/�/�/�/�/�/�/�/ d ^ h 5 W V K ? \ H  4 X J @ d ? # v F Z + Q B i   5 9 L H A X ^ B f A 9 9 T 3 O y K C < = @ S @ i f R q I a = Z 7 2 @ _ E  d m + O ^ B E i - _ / Q z Z l o + ; G  ,  �  �  \  &  �  H  \  7  �  �  $  �    �  '  5  3  r  �  �  X  �  �  �  ;  !  v  w  �  �  �  �  �  z  �  �  �  	  �  ]  �  w  �  �    M  �  �  Q  �  A  V  i    �    I        y  ,  �  �  �  !  �  :    [    h  �  �    ^  �  I  �    Z;�o;o;�o<49X��o;D��;�o�T��:�o����/�#�
��`B�#�
���w�t���t���󶼃o��P��t���/��w����h���ͼ�C����ͼ��ͼ�9X�0 Žm�h��/�T����o���C���/�49X�+�+��%��P�C��#�
��+��w��w�e`B���T���'T����o�@��'e`B�ixս��w���P�H�9�8Q콓t��e`B�L�ͽaG��u�]/���㽇+�]/���-����}�u�u���-��{��hs��F�������B%>BJ/B�IB4�8B��Bc�B+�lB ��B��B�fB�kBy�BSB��B	@�B6�B|UB{�A��IB�B�NB�	B!�B�BW�B�CB?�B0ViA��dB��B��A�(HB�"B!�B%�wB+RB#�B**uB�B4�Bl�B�+B��B$�B�B#w�BK�B�3B�KBB*B��A�VB �B�-B �B��By�B&dgB�%B+*B.��B#�=B&fB��B3HBSOB
�B^�B�BY�B�NB!��B��B��Ba�B��B
l�B#B�0BFB��B
�XB$�B��B��B4�nB�pB<B+��B�B��B��B�3B@dBG�B��B	C&B��BtBAiA�\B�,BJ&B��B!VXB�YB�pB�DB@B0@�A�pB�GB��A���B�B!3�B&ApB+-�B#�6B*:�B@EB?�BF�B=RBȒBB�VB#�B��B4B�EB>GB�A���A��B��B :�B> B��B&`�B8�B*CB.�B#��B&D�B��B��BI<B	�1B��B�SB@�B��B!��B-�B��B@
Bk�B
@B��BK�B@\B��B
�(@���AŖTA:*HAPtA�C�A2��@`M#A�(B�B?�A���A��A�O�A=��A|� AЅ@A�ÓA�*�A��AE��A�СA���AB��AW:@A�rA�kA�ҴA\�A�dAsj]A��B-Ak��A��VA�eAlw�@�a�A��AV��A���?��HA���B
$�A�ΨA���@5hBK?A��A��A���A��]A�4wA��A�z�@���A� �A�R�A5�!>�~�@�>A
�qA(�A!��C��oAvPR@��A[+uAZ�A$(�@�K~@b��@���C��B��?���@�BA�m�A��&A�7�@��@�e�Aäq@��xAņ A:�AOPA�u�A3Yf@c�5A���B EB�A���A��AƊ�A?,"A}<Aτ�ÀTÃA��'AEtAƁ�A�tADd�AWqA�{�A���A���AZ��A�qaAsKA��!B��Al��A�}_A��Ak:h@�@A���AT��A�z�?�d�A��B
;FA���A���@3��B��A�q�A�wA�}�A���A��Aφ�A�&@�6�A��A܃�A6��>��rA ��A/�A'��A �C��Av��@�|AZ�[A[ �A% �@���@\�@�Z�C��B�	?� @��A�}!Aǆ�A�z�@�_>@��HAÊ�         
                     (            
   M                                                    (   	   "   -               
   	   %            &   
   	            	                     *   %         "      	                        +   	                  +                                    +               1               !         '                           )         7   #                  +            '         )      )                           '                           +            '               #                                                                                                            '         5                                                )                           %                           '            '               #            ND��N��N���NGd\N���Nc\WN�O���N^\O���N�
Nǣ�N��hO��O��pNڇ<N���O���NǺ�O��MN?��O
6YO�k�N�V�O�O\YN��O(��O`�N���N��aPv�O�BO$_$PG��Oj��Na��N�V�O6N��/N:HsOd��O'>xN�uN�`DOW�@Oa�N��IO�=2M� O�g�OY�N�OOi{�N�^�N��N�~�O�tOQ��O��N��qNx�O{�NKK�N�<�Nx�KN�:N�	^O�>�N�G�N<�CO�O�A�NZB�N}��N�R�N�m"O��hN��O�+N��N`�  �  �  7   �  �  �  �  �  �  �  �  �  N  �  �  E    �         �  l  x  �  �    }  ?  1  �  �    �  :        �  �  �  "  �  �  �  �  {  �  �  �  �  �  �  +  �    J  �  �  D  �  �  j  m  =  �  v  �  6  0  �  a  @  *  �  l  �  N     $    T<D��<e`B<#�
<T��;ě�;�`B;�`B;ě�;�o�49X�t���o%   �o��w�D����`B�ě��o�49X�t��D����t���C��u�e`B�e`B�e`B�u��C��ě����
����ě���j��j���ͼ�9X���ě����ͽt���/��h���0 ż�����w�o�o�+�\)�#�
�\)�t��#�
��P�0 Ž�w����w�,1�,1�,1�<j�@��@��L�ͽL�ͽL�ͽL�ͽY��aG��aG��aG���o��o��O߽�1���-��{#%,.#��������������������GHUXabnnqna_UPHAGGGG��������������������������������������������������������������������������������y������������|{zqqry������������������������������������������
 ####
�����9<AHPUZacda[VUHE<599������	����������

 �����JN[gt������tg[SLHFFJrz}��������������{zr�������������������������
!--'
������TTWZ_ahmoqppnma[TTTT ;?DKIB6) �� @BCOV[a[OB>9@@@@@@@@���
	������������������������������qt|�����������tpmmqq#/<HUWUMH</##'/<HNQMPJH<:/'# UU`abnnnhaVUUUUUUUUU$*46BDNORSQROC6*����� 		����������������

�����������������������������;?HTamuz�|smaTHC<<; )6BO[`hntti[OEB6-) ����������������������� #Ubnuz{ud<0���������������������������

��������������������������������������������hht������tha]\hhhhhhW[bhioih[[UUWWWWWWWW������������������������
#%)#
��������

��������������������������������������

���������������������������������������������������������������������������������������^ilz�����������za[Y^aaekmz��������zmkeaalmrz��������~zzmkjjl�������������������������������������������"!��������������������������/0<IJPSUUPI<700,**//06;EO[hnommjh[OB60*0���������������������(*21*��������

		
����������"(0<IUW_aaZUI<4# "	
#&/7/$##
						uz{��������������zuuMOQZ[`fhokh[[[OOMMMMagt���������thg]aaaa�����������������������	)5>?;6)#����������������������������������������������������������������#/<HQ]n{���znUH</)!#������������������������������������������������������������mt������������ttnlmm )B[gg[VNB5) !)+*) ����
��������)-44/*)�����������������������������������¼������������������������U�K�J�M�Q�U�`�a�n�o�n�m�g�a�U�U�U�U�U�U�4�3�(�(�'�(�3�4�A�C�L�M�P�M�G�A�4�4�4�4���������������ʾԾվϾʾ��������������������������������	��
�	���������������׽������������(�)�(��������������������������!�%�#�!���������������������������������"�6�;�?�B�6�*���������h�g�f�h�uƁƎƁ�{�u�h�h�h�h�h�h�h�h�h�h������ƻƵƻ�������������
�
���������A�>�5�3�3�5�A�L�N�W�Z�g�i�i�g�c�Z�N�A�A�(�$��������(�-�5�A�C�C�A�<�5�(�(�z�p�n�a�U�R�O�U�\�a�c�n�y�zÇËÇÂ�{�z�A�>�4�)�+�4�A�H�M�Z�f�s�u�x�s�l�f�Z�M�A�����������̿ѿݿ������������ݿѿĿ���������ü������������������������������ìåàØÙàæìùý��������úùììììàÛÓÎÍÒØàìù��������������ùìà�t�p�h�[�O�I�O�T�[�h�tāčďēčăā�t�t�f�[�P�K�s�������������������������s�f�U�T�U�[�a�i�n�v�x�q�n�a�U�U�U�U�U�U�U�U������������������������������޾Z�M�S�W�R�P�P�Z�f�s�������������v�f�Z��ܾ׾վѾ׾ܾ�����	�	���	������²°²´����������������������������¿²���������������������������������������������������������������������������������"��	�����������	��"�$�.�4�8�.�"�	����������	���/�:�;�H�M�H�=�/�"��	��������������������������������������������������)�4�6�@�B�D�B�6�)����Ɓ�u�h�b�d�j�uƀƚƳ�������������ƧƎƁ�m�g�^�_�X�[�`�h�m�x�z�������������y�t�m�H�F�>�<�8�:�<�H�U�\�a�i�n�r�x�{�n�a�U�H�������������g�d�g�s������������������m�`�V�M�F�G�K�T�`�m�w���������������y�m�лͻʻ˻лܻ������ܻллллллл�����������������������������׾Ѿʾ¾ɾ׾����	������	�������������������������������������������L�B�L�T�Y�e�r�r�r�r�e�Y�L�L�L�L�L�L�L�L�����������������
��#�(�)�#��
���������0�*�$�����
���$�.�=�A�I�K�N�I�=�0���������������������������������������ŠřŔōŔŖŠŭŭŭŹŹŮŭŠŠŠŠŠŠ�ɺȺ����������������ɺ׺������ֺ��{�u�q�o�d�\�b�g�o�{ǈǎǔǙǘǔǍǋǈ�{¦£¦²µ¿¿¿¿²¦¦¦¦¦¦ŔōŕŠũŭŹ������������������ŹŭŠŔùðìàÕàìù������ûùùùùùùùù�m�a�T�F�>�;�D�a�m�z����������������z�m�"�!��	�������	���"�/�;�<�F�A�;�/�$�"������üù÷íù�����������������������������	����5�B�M�Y�T�E�B�5�-������������~�x�v�x�����������������������������������������������������������������h�e�a�h�tāčėĚĜĚčā�t�h�h�h�h�h�h����������(�4�A�K�I�A�4�3�(�����ܹ׹Ϲù������ùϹܹ������ ��������ܼ��������z�v�������ʼּ������ּʼ��������������!�.�1�0�.�.�!��������Ľ½����������Ľнݽ�ݽݽнĽĽĽĽĽĽ����y�u�y�����������Ľɽν̽Ľ���������ECEAE:EBECEMEPEWE\E\E\EWEPEDECECECECECEC�������������������������ĿſĿ����������û��������������ûлһܻܻܻллûûûþ������������	��"�"�"�"����	���������	�����������	�������	�	�	�	�	�	�����g�n�y�����������ĽŽӽݽ���ݽн����S�M�F�B�C�F�S�_�l�s�x��������x�l�_�S�S������������!�)�,�!����������?�4�1�4�5�@�M�Q�Y�a�f�r�x�~�r�l�f�Y�M�?F
F
FFFF$F9F=FJFcF|F~FyFuFmF`FGF1FF
��������������������������������������'�2�3�?�3�'����������~�t�t�~���������������Ǻ��������������~��������������������������������������ÓÊ�z�n�c�^�K�E�H�a�n�zÈÐÕÝÝåâÓ�����������������������������������������M�E�C�A�B�M�Y�f�r�������������r�f�Y�M�������������'�(�3�-�'�������/�,�)�/�<�H�L�Q�H�<�/�/�/�/�/�/�/�/�/�/ ; ^ : 5 U V K ? \ $  / X J 3 d " $ b D X - 8 0 l   5 9 L H : X ^ 0 ` = = 9 ^ 3 O G K > . J @ S 1 i f R q 8 a = J 6 1 ; _ E  b m 2 O ^ > E i - _ / Q z X l o ' ; G  \  �  �  \  �  �  H  \  7    �  �  �  H    '  �         r  >      �  ;  !  v  w  �  &  �  �  l  =  �  w  �  n  �  ]  �  w  �  �  �  M  �    Q  �  A  V  �    �  �  ;  �  �    y  �  �  �  �  !  �      [    +  l  �    "  �  I      Z  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  [  @  &  &  /  ?  G  N  S  [  c  m  x  �  �  �    "  -  2  6  3  -  #      �  �  �  �  �  Z  0     �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  }  m  [  E  ,    �  �  �  l  9    �  �  �  �  }  p  d  W  K  ?  9  9  9  9  9  4  .  '  !    �  �  �  �  �  �  �  �  r  e  X  J  =  0  "       �   �   �  �  �  �  �  p  M  *  
  �  �  �  �  j  D    �  �  ^  :  �  �  �  �  q  a  R  B  4  %      �  �  �  �  �  �  �  �  x  ~  �  .  \  z  �  �  �  o  T  5    �  �  F  �  �    }  �  4  O  f  {  �  �  �  �  �  �  t  T  +  �  �  �  ^  !  �  g  �  �  �  �  �  �  �  �    i  K  ,    �  �  �  �  |  T  +  N  K  H  E  D  B  J  Y  i  c  Y  J    �  �  �  {  W  2    �  �  �  �  �  }  q  b  P  <  $      �  �  �  �  �  {  d  &  �  �    <  [  |  �  �  �  �  �  �  H  �  Y  �  �  �  "  E  E  D  C  ?  :  5  .  '          �  �  �  �  �  �  �  �  �  �  �  	    �  �  �  �  �  e  B    �  �  |  D  �  �  �  �  �  �  �  �  �  �  �  �  k  L    �  �  J    �  �  �      �      �  �  �  �  �  w  T  0    �  �  �  ]  1    �            �  �  �  �  S    �  ~  (  �  u    �  S  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  >  �  �  �  �  �  �  v  j  W  D  0      �  �  �  u  $  �  5  �  2  [  f  i  e  i  b  T  B  ,    �  �  u  ;  �  �  �  E  �  �  I  Z  g  q  w  v  o  b  R  <  "    �  �  �  _  &  �  �  6  �  �  �  �  �  �  �  �  �  �  �  k  G    �  �  �  b  �  �  �  x  b  L  7  "    �  �  �  �  �  �  �  �  k  Q  0  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  r  f  X  G  5  "    �  �  �  �  �  �  �  `  "   �   �   Q  ?  8  0  $    
  �  �  �  �  �  �  t  X  9        �  �  1      �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  J  '  �  �  �  �  �  �  �  �  �  �  s  C    �  t  -  �  �  w  2  �  �  �  �  �  n  M  &  �  �  �  \    �  =  �  u  %  �  �            �  �  �  �  �  �  �  �  �  �  �  �  u  Y  >  #  U  �  �  �  �    Z  +  �  �  �  R    �  l  �  <    �  !  ,  ,    �  �  O    �  �  �  �  ~  S  $  �  �  0  �   �  �              �  �  �  �  �  �  �  g  :    �  �  g                    �  �  �  �  d  B    �  �  �  w      �  �  �  �  �  �  �  �  �  q  ^  J  6  #     �   �   �  i  �  �  �  �  �  �  �  �  �  X    �  �  B  �  �  �  v   �  �  �  �  �  �  �  �  �  z  o  c  W  J  =  2  (      �  k  �  �  �  �  �    w  o  f  ]  S  G  :  /  $  2  G  i  �  �  `  �  �  �        �  �  �    d  K  %  �  �  :  �  .   �  �  �  �  �  �  �  �  �  ~  m  ^  O  @  1      �  �  z  %  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  Y  =  }  �  �  y  d  M  4    �  �  �  �  ~  ^  >        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z    �  Q  �  �  {  q  f  l  t  u  t  n  f  X  F  /    �  �  �  �  S  !   �  �  �  �  �  �  �  y  g  R  <  %    �  �  �  �  �  R    �  k  �  �  �  �  �  �  �  �  j  N  2  )    �  �  p     �  �  �  �  	      �  �  �  �  �  n  S  7    �  �  �  �  �  g  �  �  n  R  M  �  �  �  u  a  H  %     �  �  �  ]  #  �  �  �  �  �  q  ]  H  0      �  �  �  �  �  �  �  k  O  2    �  w  M  "  �  �  �  �  Z    �  z  {  O    �  k  $  �  �  �  �    )  '      �  �  �  x  <  �  �  V  �  �  P  �  �  �  �  {  k  V  =    �  �  �  �  �  �  }  :  �  �  3  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  r  l  f  �      0  J  E  5  '      �  �  �  �  y  R  J  >  0  $  �  �  �  w  c  L  5      �  �  �  S    �  �  M  �  �  Z  M  z  �  �  {  j  Q  3    �  �  k    �  +  �  V    �  �  =  B  8  1  )    �  �  �  |  X  F  D  8  %  �  �  %  �  �  �  �  �  �  �  p  L  '    �  �  �  �  a  =      �  e   �  �  �  �    s  e  W  H  7  '      �  �  �  �  �  d  @    ?  d  j  c  T  <    �  �  s  @    �  �  �  X  �  �  �  R  H  e  z  �  �  o  J  #  �  �  �  y  M  !  �  �  }  �  )  =  =  (       �  �  �  �  �  �  �  �  z  ]  A  %  	  �  �  �  �  �  �  �  �  �  �  �  �  i  N  2    �  �  �  �  �  �  �  v  q  k  d  ^  T  H  8  %    �  �  �  q  4  �  �  P  �  �  �  �  �  �  �  �  �  �  �  o  S  6    �  �  �  t  3   �   �  ,  4  )        �  �  �  �  �  �  �  V  "  �  �  [  3  3  0      �  �  �  �  h  -  �  �  _    �  |  /  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  /  �  �  S  a  M  5    �  �  �  �  H    �  �  F  �  �  V    �  |    2  ;  %       �  �  �  f  *  �  �  V    �  -  �    �  c  !  %  )  )  '  $           �  �  �  �  �  �  i  ?    �  �  �  �  �  �  �  �  �  �  �    
          )  9  I  Y  l  \  L  <  ,      �  �  �  �  �  �  s  ]  G  -    �  �  �  �  �  �  �  �  �  �  �  �  d  >    �  �  �  E  �  �  E  N  1    �  �  �  �  �  �  X  0  �  �  �  �  �  �  l  8  �               �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    #        �  �  �  \    �  ^  �  m  �  F  �  �    �  �  �  �  �  n  I  "  �  �  �  �  \  9    �  �  �  �  T  A  -      �  �  �  |  T  '  �  �    :  �  �  \    �