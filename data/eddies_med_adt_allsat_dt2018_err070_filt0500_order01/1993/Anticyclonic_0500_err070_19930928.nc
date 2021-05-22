CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?˅�Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NI�   max       P�9*      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @E��
=p�     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
=    max       @v���Q�     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O�           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�E           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >�b      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��F   max       B,�      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��h   max       B,?      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�̸   max       C��      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�Y�   max       C��A      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NI�   max       P�O*      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_ح�   max       ?��䎊r      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >#�
      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E�Q��     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @v���Q�     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @"         max       @O�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�           �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >h   max         >h      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���$tS�   max       ?��䎊r        R<      :       
         %                                 3         	   W         V   !      "            
               �   	         
      !               
      W   D      �   	            "N�	PHH�O�Z�NR2Nc�]NA�uO��N��XNI�O�W�OiJ1OE�O}YO�|N���O��OC��P�9*N?-N��4NE�P)MaO���N&�PiΖOb�^O�-�O�F6NX�;OA�YN��Ne�SN�lO�O�N�P��P=1WN�`lN�">N�UbNfb�N�JO���O\�N�O��#O��N;��N7[PC�P!�dN���O�� Nit�N��Nt1�N���O#����㼋C��e`B�T����o��o%@  ;o;D��;�o;ě�<49X<49X<e`B<e`B<�C�<�C�<�t�<��
<�1<�j<ě�<ě�<���<���<�`B<�=+=C�=C�=\)=\)=t�=�P=�P=��=�w=�w=#�
=#�
=,1=<j=D��=H�9=H�9=P�`=]/=aG�=aG�=e`B=�%=�%=���=\=���=�h=�=�NNO[gtx������~tog[NN����#0<KW]YI.
����!-6;BOY[bh[OB60%��������������������!#/251/)#mhqt{����tmmmmmmmmmm#/<?LW^\UH</#XY[gt����{tg_[XXXXXX���������������������� 
/4<??=</#
�������������������������������������	���������������������������	")/;@A@;7/"	�����������������)+R[VB)��������������������������������������������������������������������� )5BP`a^NB)���������������������"#*/450//#""""""""�������

�������������������������������#/<DUanupeaUH<-#��������
 "������������������������������������������zwsuz~����zzzzzzzzzz�������������������� ).+)��������������������ZQ[hkqoh`[ZZZZZZZZZZ��������'%(%&#���BF[g����������tg[LGB
#%+/04/#
$)5;55565*)@@BBCIN[`fe[[UONKB@@GOQ[[hpojh[OGGGGGGGGmnvz����������znmmmm�������������������������!( ���167BKOOOKB<611111111��������������������������#)-.* ������

�����������#0<GH><0/'#)4�������)=?=5))���).6;QPJEB6��&#$')-46BKLKFB<6/)&&���������

���������

���������xppqnoz����������zxnoz������zzqnnnnnnnn����������������������������

 ����ÇÌÓÚÚ×ÓËÇÅ�z�s�n�n�l�n�r�}ÆÇ�������������������_�F�:�-����:�S�x�������Ľн۽ݽ�ӽ��������x�q�p�z���������'�3�6�9�4�3�'��������'�'�'�'�'�'�����������������������������������������"�.�;�G�;�7�.�"���"�"�"�"�"�"�"�"�"�"���'�%�#�������������������������"�/�2�2�3�0�/�(�"������"�"�"�"�"�"��������ùùöù�����������������������������	�� �"�&�$�"�	�������������������˿`�m�y���������~�y�k�`�T�G�E�6�0�=�G�V�`FF$F1F=FJFVFcFeFfFcFVFJF=F1F*F$FFFF�M�Y�f�r�����}�|�r�m�f�Y�M�@�4�)�,�3�@�M�������5�7�������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�f�m�o�w�z�����z�m�a�Y�T�L�I�K�T�Z�a�a���������Ŀ̿ѿڿѿϿȿ�������������������/�T�Z�Z�_�k�H�	�����������{���������ÓàäçìíìàÓÉÇÆÇÓÓÓÓÓÓÓ�f�s�x�x�v�s�f�Z�N�M�E�M�Z�_�f�f�f�f�f�fŇŔŚŗŔŉŇŅ�{�t�{�|ŇŇŇŇŇŇŇŇ�5�B�g¡£�t�N�B�5�!����)�5�����Ŀݿ���	����ݿѿ������������������"�/�;�E�;�6�/�"�����"�"�"�"�"�"�"�"�A�j�x�z�v�f�M�������������������о�A������'�A�L�L�@�4�'����������������'�9�6�(���������������������������ʾ��	��㾾�����������s�s��#�%�/�9�/�,�#�����#�#�#�#�#�#�#�#�#�����������#���������������ƿ������ŭŹ������������ŹűŭŬŭŭŭŭŭŭŭŭ�/�<�H�T�M�H�<�/�)�*�/�/�/�/�/�/�/�/�/�/�h�uƁƊƁ�y�u�h�g�c�h�h�h�h�h�h�h�h�h�h�(�4�C�M�Z�g�h�f�b�Z�M�@�8�3�.�����(�����ü���������������������������������āčĳĿ����#�*�-��
��������ĳĝđ�~ā�)�B�`�i�o�m�h�W�O�6�*������������)�M�O�W�Z�\�Z�M�C�A�4�-�-�0�4�A�K�M�M�M�M�����������������������������������������I�U�a�b�n�{ŀł�{�n�b�U�U�I�<�;�<�B�I�I�H�Q�U�a�b�h�a�U�H�F�@�?�H�H�H�H�H�H�H�H�
�
��������
�	��������	�
�
�
�
�������������������y�g�`�W�X�`�i�l�y���������ʾ׾�����ھ׾ʾľ����������������ɾʾϾʾþ����������������������������
��/�<�D�F�B�:�/�#�����������������
��(�A�N�Z�l�y�����z�g�Z�N�A�(�����������������������}�������������������������������r�q�r�r��������������������3�'����'�@�e�~�������������~�r�Y�L�3�:�_�x�������������l�:�-�!������� �:���ûлܻ�����������ܻлɻû���������DoD{D�D�D�D�D�D�D�D�D�D�D�D�D{DnDdD_DeDo�{ǈǔǡǤǬǡǔǈǂ�{�z�{�{�{�{�{�{�{�{�ʼּ�������������ּʼż����üʻ������������x�l�k�k�l�x����������������E7ECEPE\EiEkEuEvEuEiE\EPELECE7E+E7E7E7E7E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E~EyE� d & 7 : U B > ; l H ; [ % 9 ? A  . = 7 = + X b Z N O i = H g 9 M 0 ` t ( ; ] X ^  B 8 *  ) T m + : Y  W f - l :  K  [  c  g  �  K    �  Y    �  _  �  0  �  M  �  h  b  �  L  �  �  E  �  �  6  �  f  �  f  d  1  V    '  0  �  �    �  [  e  G  <  ;  �  n  �  $      �  �  :  z  �  k����='�<�t���o;D��;o=t�;�`B<T��<��<�h<�`B<�`B=,1<�/<�=�P=�O�<���<�/<��=�S�=�P<�h=�G�=y�#=@�=�+=,1=y�#=�P=49X=�w=H�9=8Q�=�O�>O�;=@�=49X=L��=P�`=P�`=��
=�%=T��=���=���=��=q��>n�>�=��
>�b=���=��>+>I�>�-B	��B%A�B+>B!29B��Bk�B��B	w�B�,B��B��B��By6B5�BpHA��FB��B�7B!�CBK�B#2B��B�B�B"?�B!ֵB�Bw�B��B�PA���B�9Bg~B��B��B�B
$�B�rB�"B`�B7B�B,�B�B,�Bw;BmB#�9B%��B�GB�B��BۗB��B��Be�B`�BXB	��B%G_B5�B!>BB��BS�B�\B	��B�B~,BL�B�)B>�B�BCA��hB�;B��B!��B_fB=>B��BрB� B"/�B!��B_�B�?B�B��A���B��B@�B��B�iB?�B
6�B��B�B��BF�B��B,?B��B8B��BFB#�B&6�B<�B��B>BĂB��BB�BB�B5B?�A�Gy@��GA!��?�̸A�^A`ЀA��:A��eA�	&A�hAi�C��@ُ�A���C�k�A��1Au8�A��TA�dA@�5A��A�$NAyW�A�U�A4�@�UZA�M�AP��A�r�B}�A��Aâ�B�JA9��@�SA䗽A��qA;JrB�nA�:�A�4�A��wA��AQJ�AOA��)A��K@�	X@�s�?�@�(�@��rC���BbA#@���C���C�hAȍX@��A"�1?�Y�A��Aa A�7;A��EA·�A��Ai	
C��A@�K�A��nC�gMA��'At�wA�u8A�=�AA�A�|�A�U�Av��A��;A4�:@��SA�m�AS�A�G�BMsA���A�p�BBaA:b�@��*A�zA�o�A:�LB'A�>jA�y�A�a�A��AQ��AN��A�~OA���@�@�f�?뜮@�k@�C���BF�AZ@��C��"C�2      ;   !   
         %                                 4         	   X         V   "      "   	         
         	       �   	         
      !                     X   D      �   
            #      1   !                     #            #            G            )         ;         %                        )   +                              !         -   +                           !                                                G                     '                                 )                                          -   +                     N�	O��:OY+vN%�eNc�]NA�uO�N7oSNI�OI�OiJ1OE�O}YO�<:N���N�u O��P�O*N?-N��4NE�Oڧ�O���N&�P��Ob�^O�-�O�s�NX�;O,�GN��Ne�SN�lOO�N�P��O�`]N�`lN�">N�UbNfb�N�JOe�8N�G�N�O�[�O���N'�N7[P@,P!�dN���O`[Nit�N��Nt1�N���O#��  t    �  �    (  �  )  �  u  �  �  �  e  �    �  �  8    �  	�  -  c    &  �    �  �  �  �  6  z  '  {    	  6  �  s    S  Q  ?  �  �  �  n  	V  �  v    �  �    �  	����;��
�ě��D����o��o<#�
;D��;D��<u;ě�<49X<49X<��
<e`B<�t�<�1<�1<��
<�1<�j=8Q�<ě�<���=P�`<�`B<�=t�=C�=t�=\)=\)=t�=#�
=�P=��=��=�w=#�
=#�
=,1=<j=]/=P�`=H�9=Y�=aG�=e`B=aG�=ix�=�%=�%>#�
=\=���=�h=�=�NNO[gtx������~tog[NN�� �#0<CKMI:0#
�# !(56BOTZ[]]QOB65)#��������������������!#/251/)#mhqt{����tmmmmmmmmmm"#/<HPUUURHH</.$#"b[cgt{}utgbbbbbbbbbb���������������������� 
/4<??=</#
��������������������������������������	���������������������������	"+/;>@?;5/"	������	����������5PYUB)������������������������������������������������������������������
	)5BMTURJB5
��������������������"#*/450//#""""""""��������� ����������������������������#/<DUanupeaUH<-#���������
   ����������������������������������������zwsuz~����zzzzzzzzzz�������������������� ).+)��������������������ZQ[hkqoh`[ZZZZZZZZZZ��������'%(%&#���UTX^gt����������tg[U
#%+/04/#
$)5;55565*)@@BBCIN[`fe[[UONKB@@GOQ[[hpojh[OGGGGGGGGmnvz����������znmmmm�������������������������� ����167BKOOOKB<611111111���������������������������),-,)������

������������#0<GH><0/'#�����)<?=4()2������).6;QPJEB6��&#$')-46BKLKFB<6/)&&�������	
�����������

���������xppqnoz����������zxnoz������zzqnnnnnnnn����������������������������

 ����ÇÌÓÚÚ×ÓËÇÅ�z�s�n�n�l�n�r�}ÆÇ�S�_�l�x�������������x�_�R�F�A�7�4�6�:�S�������Ľнѽн½����������y�w�x���������'�3�5�8�3�-�'�#����!�'�'�'�'�'�'�'�'�����������������������������������������"�.�;�G�;�7�.�"���"�"�"�"�"�"�"�"�"�"�����������������������������������"�/�/�1�/�"���������������������ùùöù����������������������������������������	���������������������`�m�y���������~�y�k�`�T�G�E�6�0�=�G�V�`FF$F1F=FJFVFcFeFfFcFVFJF=F1F*F$FFFF�M�Y�f�r�����}�|�r�m�f�Y�M�@�4�)�,�3�@�M����������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�d�m�n�u�z�}�~�z�m�a�Z�T�N�J�L�T�]�a�a�������¿ĿʿȿĿ�������������������������/�Q�X�W�[�T�/�	�����������~���������ÓàäçìíìàÓÉÇÆÇÓÓÓÓÓÓÓ�f�s�x�x�v�s�f�Z�N�M�E�M�Z�_�f�f�f�f�f�fŇŔŚŗŔŉŇŅ�{�t�{�|ŇŇŇŇŇŇŇŇ�5�B�N�g�z�g�N�B�5�)� ��#�-�5�����Ŀݿ���	����ݿѿ������������������"�/�;�E�;�6�/�"�����"�"�"�"�"�"�"�"�(�A�M�`�f�b�Z�M�4�����ʽƽŽҽ��(������'�A�L�L�@�4�'����������������'�9�6�(��������������������������ʾ׾���������ʾ��������������#�%�/�9�/�,�#�����#�#�#�#�#�#�#�#�#�����������������������������������ŭŹ������������ŹűŭŬŭŭŭŭŭŭŭŭ�/�<�H�T�M�H�<�/�)�*�/�/�/�/�/�/�/�/�/�/�h�uƁƊƁ�y�u�h�g�c�h�h�h�h�h�h�h�h�h�h�4�8�H�M�S�Z�a�`�Z�M�A�4�'� ���$�(�/�4�����ü���������������������������������āčĳĿ����#�*�-��
��������ĳĝđ�~ā��6�B�O�X�]�_�\�T�O�B�6�)��������M�O�W�Z�\�Z�M�C�A�4�-�-�0�4�A�K�M�M�M�M�����������������������������������������I�U�a�b�n�{ŀł�{�n�b�U�U�I�<�;�<�B�I�I�H�Q�U�a�b�h�a�U�H�F�@�?�H�H�H�H�H�H�H�H�
�
��������
�	��������	�
�
�
�
���������������������y�l�g�_�]�]�`�l�y�������ʾ׾�����׾Ӿʾɾ����������������ɾʾϾʾþ�����������������������������#�/�<�A�C�@�8�/�*�#�������������
���(�5�A�N�Z�j�w�}�r�X�N�A�5�(�����������������������~�������������������������������r�q�r�r��������������������@�e�~�������������~�r�Y�L�3�'����'�@�:�_�x�������������l�:�-�!������� �:���ûлܻ�����������ܻлɻû���������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DvDrDtD{D�D��{ǈǔǡǤǬǡǔǈǂ�{�z�{�{�{�{�{�{�{�{�ʼּ�������������ּʼż����üʻ������������x�l�k�k�l�x����������������E7ECEPE\EiEkEuEvEuEiE\EPELECE7E+E7E7E7E7E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E~EyE� d ! 9 2 U B 0 = l B ; [ %   ? D  1 = 7 = ) X b G N O L = D g 9 M 2 ` t  ; ] X ^  A 4 *  $ M m + : Y  W f - l :  K  �  �  ?  �  K  `  M  Y  �  �  _  �  Y  �  3  5  	  b  �  L  �  �  E  �  �  6  D  f  u  f  d  1  �    '  �  �  �    �  [  �    <  	  �  B  �        �  �  :  z  �  k  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  >h  t  o  h  _  S  E  2    	  �  �  �  �  �  ~  a  I  G  5      F  ~  �  �  �          �  �  �  s  3  �  t    ~   �  U  �  �  �  �  �  �  �  �  q  N  4    �  �  A  �  �  @  �  u  �  �  �  �  �  �  �  {  m  ]  I  1       �  �  �  �  {              
    �  �  �  �  �  �  �  �  �  k  V  A  (  ,  0  3  7  ;  ?  ?  =  :  8  6  3  )     �   �   �   �   �  �    n  �  �  �  �  �  �  �  �  J    �  w    �  <  �  �  %  &  '  (  (                             �  �  �  |  r  a  Q  G  :  '      �  �  �  �  �  `  ?    �  {       4  I  Z  h  p  u  t  l  Z  =    �  �  Q  �  �  Z  	  �  �  �  �  �  f  D    �  �  �  V    �  �  2  �  �  8  4  �  �  �  �  �  �  �  �  [  %  �  �  l  )  �  �  R    �  /  �  �  �  �  �  �  �  ~  d  F  '    �  �  �  r  >    �  �  4  D  N  Z  d  `  P  ;  !  �  �  �  �  s  F    �  j  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  B  	  �  w  (  �             �  �  �  �  �  �  �  g  -  �  �  :  �  5   S  �  �  �  �  �  �  �  �  �  �  j  F    �  �  n     �  D  �  �  �  �  �  �  �  �  �  w  H    �  �  X    �  a    �   �  8  )      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      	     �  �  �  �  �  �  �  �  �  �  �  �  z  D    �  �  �  �  �  �  �  �  �  q  R  0    �  �  �  i  >    �  �  �  	  	w  	�  	�  	�  	�  	�  	�  	�  	O  	  �  ?  �  "  v  �  r  �  -    	  �  �  �  �  �    e  M  ;  0  4    �  �  z     �  c  a  _  \  Z  X  U  R  O  L  K  M  N  P  R  >  #  	  �  �    \  �  �  �        �  �  p  3  �  �  9  �  '  d  x  �  &            �  �  �  �  �  �  Z  "  �  �  `  �      �  �  �  �  �  �  �  �  {  j  W  =    �  �  �  �  W      �  �    �  �  �  �  �  �  �  �  g  3  �  �  �  R    �    �  �  �  �  �  �  �  �  �  y  o  b  R  /    �  �  �  �  �  �  �  �  �  �  �  �  l  G    �  �  [    �  *  �  �  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  b  P  >  ,      �  �  �  �  �  ~  z  s  k  a  V  G  4    �  �  �  �  n  G  6  /  (  "          �  �  �  �  �  �  �  �  �  �  �    m  q  u  z  z  z  v  n  c  Q  >  *         �  �  �  �  2  '                �  �  �  �  �  �  v  ]  D  *    �  {  n  W  X  /  	    �  �  �  �  a  :  #    �  �  8  �  ]  �  �  �  2  �  �        �  k  �  ;  |  �  �  +  	y  G  �  	            �  �  �  �  �  �  �  �  �  �  �  �  �  �  6  .  %             �  �  �  �  �  �  �  �  �  �  r  `  �  �  �  �  �  �  �  �  m  X  @  (    �  �  �  �    $  &  s  o  j  c  [  Q  G  >  7  +      �  �  �  |  V  0    �    �  �  �  �  �  �  �  �  z  c  K  0    �  �  �  |  K      +  =  N  Q  L  B  !      �  �  �  �  �  �  [  "  �  �  K  O  Q  Q  N  F  =  4  )      �  �  �  �  j  +  �  �  X  ?  2  &         �  �  �  �  �  �  �  |  d  K  3       �  �  �  �  �  �  t  Y  ;    �  �  �  �  Q    �  �  �  �  �  �  �  �  �  �  �  �  �  d  :  	  �  �  >  �  q    �  {  \  �  �  �  �  �  �  �  �  j  M  ,    �  �  �  h  =    �  �  n  m  k  j  i  e  [  P  F  <  7  7  6  6  6  +        �  	V  	A  	+  		  	  	  	&  	"  	  �  �  p  h  +  �  u  �  {  �    �  �  �  �  �  c  !  �  �  �  {  X  $  �  b  �    &  �  +  v  j  ]  M  5      �  �  �  �  �  m  C    F  �  �  �  �  :  �  a      �  �    �  �  v  �  A  /  �  '    �  �  �  �  �  j  B    �  �  �  �  �  �  p  J  "  �  �  �  ^  (  �  �  �  �  �  �  z  c  K  1    �  �  �  �  b  '  �  �  e       �  �  �  �  u  T  1    �  �  �  k  :    �  r  �  �  $  �  ^  .  �  �  �  `    �  R  �  �  c  �  �  B  �  g  �  L  	�  	�  	b  	3  	  �  �  x  2  �  �  *  �  N  �  7  I  ?  4  &