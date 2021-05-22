CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N*�   max       Ppx�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =��      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F��G�{     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vh�\)     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >�$�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��L   max       B3�t      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��d   max       B3��      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�s      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�e      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N*�   max       O���      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����ݗ�   max       ?㴢3��      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       >+      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F��G�{     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vh(�\     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q            l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�@          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�B����     �  N�   
         	      0      1         �      /   	      '   
                     
            0                            r   <            �      $   !            -   )   A   A      N�d�N��KON`N�7Oc��PM[�O�?rPpx�O��Op��Pc��N2��P"��O|�|N�٢O�*�NߓN�	�N�z}O~�qN*�N$�N��Nyq�O���N� �N�jLPA��Nr�\O7�N��rO!�O?�N��O&��O���P̄P3��NfO�P�NM��Oҙ�N��NO���OMSKN�-�O"Q�N5�O���O�2zOϬ�O�I�N�	�N%3t��#�
��`B��o��o;o;�o;ě�;�`B<#�
<49X<49X<D��<D��<T��<T��<T��<e`B<u<�C�<�t�<�t�<�t�<���<��
<�1<���<�/<�/<�/<�<�<�=o=+=+=\)=t�=��='�=49X=H�9=L��=T��=T��=Y�=m�h=m�h=q��=��=��P=�{=�{=��}�����������)6B>6))$&-<IUWadeebXUIE<0)��������������������)5BOY][NB:50)mebcl��������������m���������	�����������10*)������������� �����	"./;DHKOOMH;/"	A=:<F[t���������t[NA��������������������/1BNg��������tg[WG</������������������#'$
��������������

������##0480$############,0<BIU\_]UPI<80*#&'//8<CHTHG<:/&&&&&&`cmz���������ztmhfc`���������������������#/:80/#�����������������������������
��������hls~������������xuhh)5775)aUH/#����#<MTW_ma��
#%)&#
�������DBCHN[bgjtwz|ztsgVND868<HTU_acbaZUTHD<88eddehot���������tjheshnpttw��������}wwts48<GHLUXUHF<44444444���������������������������
#,.*#
����zrz���������������{z�����)1/)�����������


����������������!""$�������������������������������
������_\]fhtvxxvth________
)5BFKJHD5)������
$''#!
����"#$./<HTPHHIIH</,%#"��������������������ww����������wwwwwwww����������� ��������93028;>DHTahuxxq]H;9��|{�����������������������
&+.-(#
��)06@>966)MNOW[hjih_[OMMMMMMMM��������������߹�������������àäààÙÜÓÇ�ÃÅÇÓÕàààààà�����������ŻŻû����������x�v�v�x�����²µ¿��������¿¾²¯©²²²²²²²²��"�.�:�I�R�N�G�;�.�,�"����	�����A�N�Z���������������������s�A�5�*�&�+�A�������ûлۻۻٻλû��������y�{����������5�N�S�c�f�]�N�����ݿ��������Ŀݿ��������	������	���������������������a�m�t�z�z���z�u�m�a�H�;�3�1�<�?�D�H�T�a�������6�P�^�a�Z�O�6��������������������������������������|������������������5�B�R�N�`�e�`�W�B�5��������������5�����������������������������������������z�����������z�m�g�j�m�t�z�z�z�z�z�z�z�z�4�A�O�Z�^�W�D�4�(����������������4�нݽ�����ݽѽнннннннннннм���������������������v�r�p�p�r�z����������������������������������������������������������������ƳƮƮƳƽ��������ƳƷƷƳƧƟƚƐƘƚƧƬƳƳƳƳƳƳƳƳ�uƁƈƁƁ�x�u�j�h�e�h�s�u�u�u�u�u�u�u�u������������������������m�z���������z�m�g�h�m�m�m�m�m�m�m�m�m�m�����������������������q�d�d�g�n�s�|����ʾ׾������������׾Ծ˾ľ��������Z�c�f�n�p�p�f�Z�X�N�P�V�Z�Z�Z�Z�Z�Z�Z�Z����������4����������~�s�f�M�4��������������߼���������𾘾��������Ⱦʾ˾ʾþ��������������������������
���������������������������뽒�������Žн۽׽нĽ������������}����������(�4�A�G�M�Q�I�A�4�(�����������T�a�b�a�a�T�K�H�@�B�H�O�T�T�T�T�T�T�T�T�������������¿������������y�x�y���������a�m�r�n�d�l�y����y�`�;�"����)�1�Q�a�ܹ����	��������Ϲù����������Ϲ��/�;�T�a�m�m�i�W�/�� ����������������/��#�/�/�1�0�/�#�������������"�.�;�G�T�V�]�\�T�H�.��	��������	�"Ó×ÓÏÇÃ�z�n�j�k�n�zÇÎÓÓÓÓÓÓD�D�D�D�D�D�D�D�D�D�D�D�DoDPD>D@DIDVDbD��!�-�:�F�L�L�F�:�-�!���!�!�!�!�!�!�!�!�������
�#�-�2�3�3�0��
���������������غ����������ȺȺ������������~�z�r�}������¦§²¿������¿²¦¦¥¦àèìù��������������ùìáàØÖÜÛà���������������~�s�s�s�z���������������������<�H�U�[�S�H�1��������������������hāčĦĿ�������������ĿĳĦĚā�b�e�h�r�����������ϼԼռʼ������l�X�Q�Q�Y�f�rEuE�E�E�E�E�E�E�E�E�E�E�EuEiEfEeEgEiEsEu�~�������������������~�~�v�{�~�~�~�~�~�~�Y�f�f�r�w�w�r�f�c�Y�U�R�Y�Y�Y�Y�Y�Y�Y�Y $ � $ p E N + B N -  W 8 2 J @ / b W ' { V 4 & G v K R [  @ M ^ q 1 N  F 8 ; l = @ +   / / R = ^ ? , ; V  �  �  �  W  �  �  6  �  d  �  �  f  �  �  �  D  .    �  �  c  >  �  v  M  u  �  �  �  �  �  r  �  a  i  {  a  A  w  <  �  �  �  �  �    p  n  Z  �  �  '  �  Q�o�D��<#�
;��
<49X=H�9=C�=aG�<�/<�/>�$�<u=m�h<��
<���=T��<�9X<�1<�t�=49X<��
<��
<�`B<�h=�w<�`B<�=���<��=�w=L��=D��=<j=C�=0 �=��>�P=\=8Q�=�hs=}�>`A�=u=�9X=�1=��=���=�%=��=�l�>V>�u=Ƨ�=�;dBbvBB&��B�yB&oB � B"�=B��B�A��LB	�BGsB	�B�B��B#~�B%~}B&`B��B B'�B��B�B�B"e�B3�tBv�BB$�B�XBPcB`�BʅB�B)�B%�BTwB\�BR�B�lB��B��B��B��B$3zB�Bq�B
��Bv�A�mB�B�pBGKB7jBE�B��B&��B >vB��B ?�B#'�B�B�[A��dB	�YB{�B
;�B�B�_B#J�B%�;B&L)B��B 6B@�B��B��B#B"�cB3��B]hB@_B%%B	&�B=�B?�B��B�qBAB@ BZ�B�B?�B��B�wB>�BBHUB#��B�BKYB
��B��A�_B<�B�\B@�B@�?H]A��j@���A�mhA`�+A�8�@�~�A��cA�_�A��A�TA�_�A�vLA��A��\A7A+��@�sA���B��B�BB�^A���A��1@�RAS��A?�[A;G*ABAAK�sA�/�A#
A7wA�#cAr��Ad��>��A�Y�A�`A`�A��C���@w�A�1@�A���A�F�A��A�-A��@��?C�s@�n@ݧ�?J��Aɵ�@���A�~dAa�A��d@��A�T�A��&A���AՀA�{�A��A�m�A���A7��A+�@�	ZA�NB�8B�WB��A���A��@�;ATWA?uA= �A�
AK?�A�waA#H�A68A�|�At�Ah��>��A�|�A�%=Aa�AȃC��@x�A�y�@�UA��EÁ(A���A���A��0@��C�e@	�@�.�            
      0      2         �      /   	      (   
                                 1                        !   r   <            �      %   !            -   )   B   A                        /      1         -      /                                             .                        )   %   3            !                     #   %   !                           !                                                                  %                        )      '            !                        %            N��N��KO;�(N�7Oc��O�QROWZgO��fO��Op��O�F�N2��OjX�O|�|N�٢O�CWNߓN�	�N�z}O+PN*�N$�N��Nyq�O���N� �N�jLO�pFNr�\O"��N�O
�@N�R�N��O&��O���O��sO��DNfOT��NM��O�L�N��NO~��OMSKN�-�N�FMN5�O��.O�2zOs$wORߤN�	�N%3�  E  7  q  �  5  �  �  X  �  �  �  �  e  �  �  �    |  -    �  �  M  o  �  �  8  �  n  �  �  S  �    �  d  �  .  �  �  F    �  x  O  �  �    s  �  		  v  J  ���C��#�
�ě���o��o<��
<#�
<�h;�`B<#�
>+<49X=o<D��<T��<�o<T��<e`B<u<ě�<�t�<�t�<�t�<���<��
<�1<���=�w<�/<�`B=C�=o=\)=o=+=+=�O�=H�9=��=<j=49X=L��=L��=q��=T��=Y�=}�=m�h=�o=��=�j=ě�=�{=���������������������)6B>6)'(/<IU_bcddbaVUI<10'��������������������)5BOY][NB:50)lmuz������������zuol����� ��������������������������� �����	"./;DHKOOMH;/"	USTY_gt���������tg[U��������������������YVW]et���������tge[Y������������������#'$
��������������

�������##0480$############,0<BIU\_]UPI<80*#&'//8<CHTHG<:/&&&&&&mjhhmvz���������zsmm���������������������#/:80/#�����������������������������
��������hls~������������xuhh)5775)��#/<HMPPS]U</	����
#%)&#
�������GDEKN[gtuyzxtng][NGG:7:<HNU[^UH<::::::::gddfhirt��������tlhgostvz���������ttoooo48<GHLUXUHF<44444444���������������������������
#,.*#
������������������������������#',+)���������


����������������������������������������������
������_\]fhtvxxvth________		$5BDFECB>5)	������
$''#!
����"#$./<HTPHHIIH</,%#"��������������������ww����������wwwwwwww��������������������93028;>DHTahuxxq]H;9��������������������������
#'*'#
����)06@>966)MNOW[hjih_[OMMMMMMMM����������������������������àäààÙÜÓÇ�ÃÅÇÓÕàààààà���������Ļû��������������x�w�x��������²µ¿��������¿¾²¯©²²²²²²²²��"�.�:�I�R�N�G�;�.�,�"����	�����s�������������������s�Z�I�A�:�=�H�N�Z�s�ûлллǻû�������������������������������(�5�A�G�M�Q�G�A�5�(���������������	������	���������������������a�m�t�z�z���z�u�m�a�H�;�3�1�<�?�D�H�T�a����)�6�@�G�I�G�B�6�)����������������������������������|��������������������)�5�B�M�Q�K�B�7�5�)�������������������������������������������������z�����������z�m�g�j�m�t�z�z�z�z�z�z�z�z�4�A�M�X�\�T�M�A�4�(����������	���4�нݽ�����ݽѽнннннннннннм���������������������v�r�p�p�r�z���������������������������������������������������������������������������ƺƸ������ƳƷƷƳƧƟƚƐƘƚƧƬƳƳƳƳƳƳƳƳ�uƁƈƁƁ�x�u�j�h�e�h�s�u�u�u�u�u�u�u�u������������������������m�z���������z�m�g�h�m�m�m�m�m�m�m�m�m�m�����������������������q�d�d�g�n�s�|����ʾ׾������������׾Ծ˾ľ��������Z�c�f�n�p�p�f�Z�X�N�P�V�Z�Z�Z�Z�Z�Z�Z�Z�s�}���|�{�w�m�f�Z�M�4����
��A�M�Z�s������������߼���������𾘾��������ƾɾ������������������������������������������������������������뽒���������ĽнֽѽнĽ������������������(�/�4�A�I�B�A�4�(�����������T�a�b�a�a�T�K�H�@�B�H�O�T�T�T�T�T�T�T�T�������������¿������������y�x�y���������a�m�r�n�d�l�y����y�`�;�"����)�1�Q�a���ùϹܹ����� ������ܹϹù������������;�H�T�`�e�f�a�\�T�H�/����������$�3�;��#�/�/�1�0�/�#�������������.�;�?�G�R�X�U�T�G�;�.��	���������"�.Ó×ÓÏÇÃ�z�n�j�k�n�zÇÎÓÓÓÓÓÓD�D�D�D�D�D�D�D�D�D�D�D�DoDQD?D@DIDVDbD��!�-�:�F�L�L�F�:�-�!���!�!�!�!�!�!�!�!�����
��#�&�#���
�������������������店���������ȺȺ������������~�z�r�}������¦§²¿������¿²¦¦¥¦ìùÿ��������������ùìçàÜÚàáìì���������������~�s�s�s�z�������������������
�#�<�H�T�W�P�H�/�#��
���������������hāčĦĿ�������������ĿĳĦĚā�b�e�h�r��������������Ƽ̼ɼ��������r�f�_�m�rE�E�E�E�E�E�E�E�E�E�E�E�EuEmEiEkEqEuE}E��~�������������������~�~�v�{�~�~�~�~�~�~�Y�f�f�r�w�w�r�f�c�Y�U�R�Y�Y�Y�Y�Y�Y�Y�Y $ �   p E N (  N -   W ' 2 J = / b W   { V 4 & G v K @ [  7 M : q 1 N  B 8 8 l = @    /   R 3 ^ L $ ; V  �  �  �  W  �  �  �  S  d  �  I  f  �  �  �  �  .    �  h  c  >  �  v  M  u  �  >  �  Y  �  B  �  a  i  {  O  L  w  �  �  �  �  �  �      n  �  �  �  �  �  Q  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  E  E  E  C  @  :  3  (      �  �  �  �  w  ?  �  x   �   j  7  *      �  �  �  �  �  �  �  �  �  �  �  �  �  j  C    n  q  k  a  T  E  2      �  �  �  �  s  N  )    �  �  �  �  �  �  �  �  �        ?  �  �  �  �  �  �  �  �  �  �  5  3  +      �  �  �  �  v  V  8  &  	  �  �  �  o  P  %  �    R  i  y  �  �  �  �  �  �  q  J    �  �  ,  �      v  �  �  �  �  �  x  j  _  `  Z  F  "  �  �  �  P    �  +  	  #  d  �  �  �    B  U  W  K  '  �  �  @  �  Y  �  N  b  �  �  �  �  �  �  �  �  �  t  [  ?    �  �  �  �  �  �  2  �  �  �  �  �  �  �  �  �  �  �  w  R    �  �  *  �  2   �  �  �  v  �  �  �  ,  �  �  �  �  ~  �    �  �  f  �  	�  �  �  �  �  �  �  �  �  �  �  �  p  ^  L  :  '    �  �  �  �  ^  �  �  �     :  N  ^  e  `  O  2    �  �  G  �  h  �  �  �  �  �    o  `  O  ?  -    
  �  �  �  �  �  �  �  w  a  �  �  �  �  �  �  �  �  �  �  u  g  Y  I  4    �  �  7   �  �  �  �  �  �  �  �  s  K    �  �  u  &  �  ]  �  R  �  �        	    �  �  �  �  �  �  �  �  �  �  �  l  V  ;  !  |  j  X  G  6  &              �  �  �  �  �  �  n  M  -  &             �  �  �  �  �  �  �  �  �  �  �  t  e  �  �  �  	          �  �  �  �  k  9  �  �  1  �  F  �  �  �  ~  w  q  k  d  ^  X  Q  P  T  X  \  `  d  i  m  q  u  �  {  u  n  g  `  Y  R  K  D  :  ,        �  �  �  �  �  M  B  6  )    
  �  �  �  �  �  d  @    �  �  �  �  _  6  o  g  ^  U  L  A  7  +        �  �  �  �  �  u  Y  ]  e  �  �  �  �  �  �  �  r  X  :      %      �  �  ;  �  �  �  �  �  |  n  `  S  H  =  /  !    �  �  �  �  �  a  .   �  8  :  <  >  >  5  -  $      �  �  �  �  �  �  t  Q  -  	  �  �  �  �  �  �  �  �  �  j  3  �  �  =  �  s      h   �  n  e  ]  T  K  B  7  ,  "        �  �  �  �  �      !  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  V  <    �  d  P  �  �  �  �  �  �  �  �  b  ;    �  �  l    �  Z  �  �  L  P  S  R  L  ?  -    �  �  �  �  �  w  =  �  �  3  �  i  b  �  �  �  �  �  �  �  �  |  Z  3    �  �  �  k  ;    �    �  �  �  �  �  �  �  �  �  �  �  �  �  t  h  \  P  D  8  �  �  �  �  �  �  t  e  X  L  8    �  �  �  |  R  *   �   �  d  /  �  �  �  �  �  K  I  D  �  �  �  �  X    �    |  �    �  �  5  b  }  �  w  Y    �  a  �  7  
s  	�  s  $    �  |  �  �    .  -  !    �  �  �  h  #  �  T  �  g  �  �  q  �  �  �  �  u  i  ]  Q  F  >  6  /  (  "          �  �  �  �  �  �  �  �  �  ~  f  R  >  "  �  �  �  2  �  @  �    F  �  �  �  �  �  �  �  i  A    �  �  �  m  <    �  �  k    �  (  J  \  \  N  1    �  �    _  J    �  1  ~  9  	s  �  �    y  r  k  a  R  ?  %  	  �  �  �  U    �  �  i  )  U  f  n  t  x  s  e  T  ?  #    �  �  L  �  l  �  o  �   {  O  E  ;  1  $      �  �  �  �  ]    �  �  I  �  �    �  �  �  �  �  }  X  *  �  �  x  :  �  1  >  �  �  5  �    b  �  �  �  �  �  �  i  9    �  �  p  ;    �  �  >  �  ;  m    
  �  �  �  �  �  �  �  c  A     �  �  �  h  4      �   �    /  r  i  X  <    �  �  {  _  A    �  �  .  9  �  �  `  �  �  i  E  %  
  �  �  �  �    <  �  �  ^  �  J  �  �  �  =  {  �  �  	  	  	  �  �  �  g  5  �  �  x  
  �  �    B    c  m  t  r  g  E    
�  
�  
A  	�  	k  �  \  �  �  �  V  <  J  ?  2  "    �  �  �  �  �  �  s  `  N  <  *      �  �  �  �  �  �    }  {  b  =    �  �  �  H    �  �  ;   �   �