CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�t�j~��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N6/   max       P�!�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �C�   max       =�;d      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>u\(�   max       @F�
=p�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vf�G�{     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @O�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @��          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       >���      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��l   max       B/_�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��"   max       B/�X      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =5)�   max       C��      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =xw�   max       C��      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         0      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N6/   max       PО      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���,<�   max       ?��&��IR      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �C�   max       >5?}      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�
=p�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @ve��R     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @M@           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @˼        max       @���          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?���n��     0  O�            
      #   #   :      /   	         *                           :         4   g   C            C        0   3   )   #   !            &   �   M         "   E         :      	   �N1�UNF��N��ND��N�%�PtGOj��P]��N]��P�O$Q�OV�.OFC
Oҧ�O�S�N�SfPX�NQ2BOL%VN�\�O,z�O��~P[N�9�O�O�*�PS�~OB�9NPOl{VNY��P�!�O��UN�QP`��O��PОO�OYO��fN6/O�<8O���O��PP+��P7�N�EWO�IO�DfOV�O9s*N�t�O��5N�aNH�hO�۽C��D���t��t��o���
�o;ě�;ě�<o<e`B<e`B<�C�<�t�<���<��
<�1<�9X<�9X<ě�<ě�<���<�/<�/<�`B<�`B<��=o=+=\)=\)=�P=�P=�w=49X=49X=8Q�=@�=@�=@�=H�9=H�9=L��=e`B=e`B=ix�=m�h=m�h=q��=�O�=�t�=��P=��T=�E�=�;d--/<HOSH</----------��������������������()-57BIN[gmgd[NB5)((zx����������zzzzzzzz����������������������"/;DQROE;/"	����#/<U[abaUH</*#VU[gt�������������fV��������������������������
')(%
���������������������������������������������#/<COLHC<4/#�������������������&..)#
���1*06<BOORPOJB6111111]el���������������t]y~������������yyyyyy��������������������
�
#050+#






""$((.0<IU[[XSI<00#"�����)9MSQKB<)���������
�����|�����������������||ooty~�������������zo��������������������hgiez������������}xh#/<HNURKH</##�������������������b^a`bjt�����������tb&*-3*(����0LVVOOLB)	������)27971.3)��������������������-,1B[gt��������gNB8-���������������z����������������~{z�������������������������$-00)$�����������������������������������������������������������������������������32AOTm���������zmT;3������)2>A=5)���vv������������vvvvvv�����)5<<BD@5)�#/<HOY_cea_U</!�������
! 
���wz����������������|wnwxz����������|znnnn����).9GFC@)��YU[amz{~}zsmbaYYYYYYWRRZ[htohe\[WWWWWWWW���������
�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D쿟�����ĿпĿĿ������������������������������ʾ־׾پ׾ϾҾѾʾ¾��������������������¹ùŹĹù�������������������������������(�0�(�����������������T�a�m�w�x�r�a�;�/�"�	������	�&�=�H�Q�T���������������������������������/�?�N�T�V�K�I�;�/�"���������������������������������������������������������)�5�B�N�`�g�t��z�[�B�)������������)�a�m���������������z�s�m�a�T�P�T�Z�[�_�a�"�/�;�B�I�H�E�?�;�/�"��	��������	��"�����������������������������������}�|���<�H�U�a�z�~�z�l�a�U�H�<�*�!��"�#�)�/�<�������ûлۻڻл��������x�l�e�]�]�f�q���z���������������������z�u�v�z�z�z�z�z�z�4�M�Z�s�����t�f�Z�M�A�4��������4�T�`�a�`�d�`�T�N�G�;�7�;�G�J�T�T�T�T�T�T�#�(�!���	���׾ʾľ¾ʾ;׾���	�!�#��'�4�:�:�@�@�4�.�'�#����������M�Y�f�r���������������r�f�Y�T�M�G�A�M���!�-�:�I�K�>�-��������������	��"�)�/�0�.�)��	��׾����������׾�	���������������������~�~�~���������������O�\�h�uƁƎƎƒƎƂƁ�u�o�h�\�X�P�O�L�O�Ϲܹ������������ܹ����������ùϺ����!�;�<�%�� ����ֺ̺ʺ�����������`�m�y���������������y�m�`�]�T�Q�T�T�_�`�`�m�x�y�|�z�y�m�i�f�`�`�`�`�`�`�`�`�`�`���'�3�@�Y�f�i�j�e�Y�L�@�3�'������������������������{�y�s�y�������������������(�0�������Ƨ�\�*�����CƁƟƳ�̺Y�e�r�~�������������~�r�e�Y�L�A�E�L�U�YÇÓàäìðôìàÛÓËÇ�z�y�y�zÀÇÇ���6�J�U�U�Q�B�6���������������������)�6�B�O�[�h�|ăąā�t�h�O�B�:�/�&�!�!�)���������������������g�A�(�!�$�/�A�Z�����	��"�/�;�T�a�k�t�z�|�z�o�a�D�,����	�����������������������������������~�~���������������������������������������������ѿ�����)�5�4�*�����ݿͿĿ������Ľн������!�#�#����нĽ��������Ľ�����������������������y�n�b�_�`�l�y��čĚĳĿ������������ĳĚčā�q�i�i�s�|č���
�0�<�I�Q�X�W�Q�I�0������������������<�H�R�U�Z�U�M�H�<�/�/�/�/�6�<�<�<�<�<�<�������ĿοͿĿ��������}�y�m�a�`�f�m�{���A�M�Z�g�n�p�l�f�Z�M�A�4�/�(�"���(�4�AEiEuE�E�E�E�E�E�E�E�E�E�E�EuEoEiEgEgEeEi�������ʼ̼ʼǼ�������r�f�d�r��������������(�*�(�#����������������Y�f�r�������������f�Y�M�4�1�0�4�<�M�YŭŹ����������ŹŭŧŠŜŠšŭŭŭŭŭŭ���'�4�4�9�4�'������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDjDjDqD{ Z P o E ^ 2 D , \ ( / 1 ( / N ) + M ~ b P ^ j ; J : @ % l L 1 t %  " E : ^ G H g , > S  B H   " Y '   X     m  w  F  r  �  l    �  �    p  �  �  �  �  �  �    +  �  �  �  �  �  N  �  �  �  e  �  d      �  �  [  7  �  �    �  �  z  <  �  �  �  2  �  �  �  {  �  z  )���t��ě�:�o��o<�h=o=�%<t�=]/<�9X<�1=+=u=@�=o=8Q�<���<�<�h=t�=Y�=�{=C�=,1=���>1'=ȴ9=\)=y�#=��=���=�7L=��>���=\=�{=��=��
=P�`=���=�C�=�9X>N�>+=�C�=��-=�^5>o=�{=�Q�>+=�v�=Ƨ�>�VB�]BϯB�B�B��A��lB�hB
�Bi�B4�B6�B�BDBsB$�B�BBxB/�B!NB$��B&K�BoB �B��B F�B��B�?BZ�B-MbB��B/_�B�&B��B!�B	@4BNPB�7B�LBu�B��B�lB"&B,��A�q�Bl�B!�BVB��B��B��B��B�<A�MPBy>B��B�.B�B�;B:�B�NA��"B��B
�+B��B>�B`FB��B��B�	B#��B3/B
�OB
�B!A_B$��B&ApB�BB�By�B �%BͿB!�BE�B-1�B��B/�XB�)B}zB"?}B	>pBAB@�B��B��B�TBDFB"B,��A���B�BBF�B��B�NB�BǝB��B�SA��BR*B��C�> AvJjAO��=5)�A�[A���A��]A�^gA��A�PA�k�A�~�A���A���@��A���A:�Af�SAU��@�O�@�-�@fdAU��@�<B>�KI@B�@Ak��Aj��?�BAo�#B�P?��"Aʉ�A�PA�״A���A��A���A�r�A��A/P7AfA�2A�jjA�ӬAq�A;��C��@��A2Pm@�qA�?@�<C���C�<�Au��AP��=xw�A�}aA�o�A҇�A���A��A���A�1A�L�A��jAŀE@���A�ZA; �AgCAT��@��@�H�@oElAV�A@|�B�'>��]@D%Ak�QAi��?ț�Ao �B;�?�T�AʂA�|�AهA���A�pA�|�A��A��:A/öA>A�}+A�y�AèAq�A;��C��@���A2�@�m-A�n@�YxC�ʜ                  #   #   :      /   	         +               	            ;         4   g   C            D        0   4   )   $   "            '   �   M         "   F         ;      	   �                  #      1      '            !   #      '                  )         !   1               E         -      5   !         %      !   )   )                                                      '      '                     '                  !         !                  7               5            %            )                              N1�UNF��N��ND��N�%�O�~N�S�P�*N]��O�-�O
�KOV�.O4�O�0Om��N]S\P	��NQ2BN꭫N�\�N��O���O�{TN�9�O�O�*�O��aO2��NPO_�NY��Pb�O�m�N�QO���O}�!PОO/V�O��fN6/O�<8OM�N��OU$�P2	SNk�{O�IOo�%O�*O9s*N�t�Om�IN�aNH�hOAآ  |    c  ^  E  h  ?  �  $  �  v  �  T  5  �  �  I  �    ^  �    �  �  �  "  
u  �  �  �  K  "    �  �  �  +    {  �      �  {  	
  2  (  �  i    �  4  7  �  ��C��D���t��t��o;ě�<t�<�9X;ě�<e`B<u<e`B<�t�<���<���<�9X<�9X<�9X<ě�<ě�<���<�h=#�
<�/<�`B<�`B=��w=t�=+=t�=\)=T��=��=�w>5?}=H�9=8Q�=m�h=@�=@�=H�9=e`B=�+=��m=ix�=m�h=m�h=}�=��=�O�=�t�=�-=��T=�E�>��--/<HOSH</----------��������������������()-57BIN[gmgd[NB5)((zx����������zzzzzzzz�������������������� �	"/;=@DJIF;/" "#/<AHOUURH@<8/($#d`ct�������������{rd�������������������������
!$# 
�����������������������������������������������#/<AHLJHA<2/#�������������������&&#
���3,26BNOOOHB633333333`fm��������������tj`y~������������yyyyyy��������������������
�
#050+#






%'**02<IUVVUUOI?<0%%���),@JQOGB?)�������

�������|�����������������||ooty~�������������zo����������������������������������������#/<HLSQIH</)#�������������������dbb`bkt�����������td&*-3*(�����7INOHD@5)!����)16970.1)��������������������A@EN[gt�������tg[NFA���������	 �������z����������������~{z�������������������������$-00)$�����������������������������������������������������������������������������������^YX_lmz��������zqma^������)2>A=5)���xx����������xxxxxxxx�����)5<<BD@5)�#(/<HV[]\UH</%#������

�����wz����������������|wnwxz����������|znnnn����)18=<8.)�YU[amz{~}zsmbaYYYYYYWRRZ[htohe\[WWWWWWWW�������� 
������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D쿟�����ĿпĿĿ������������������������������ʾ־׾پ׾ϾҾѾʾ¾��������������������¹ùŹĹù�������������������������������(�0�(�����������������H�T�a�f�l�o�o�e�a�T�H�;�/�"��
���/�H�������	�������������������������	�"�/�?�G�H�B�/�"��������������������	�����������������������������������������5�N�[�c�g�u�n�[�B�)������������)�5�m������������������z�v�m�d�a�^�]�a�b�m�"�/�;�B�I�H�E�?�;�/�"��	��������	��"���������������������������������������<�H�U�a�n�v�{�u�g�a�U�H�<�4�)�$�*�*�/�<���������ûͻͻû��������x�s�l�g�f�m�x���z�����������������z�v�x�z�z�z�z�z�z�z�z�4�M�Z�s����s�f�X�M�4�(��	��
���(�4�T�`�a�`�d�`�T�N�G�;�7�;�G�J�T�T�T�T�T�T�׾����	��	�	�������׾ʾǾƾʾԾ׼�'�4�:�:�@�@�4�.�'�#����������Y�f�r���������������r�r�f�^�Y�X�P�Y�Y���-�:�F�I�C�<�:�!��������������	��"�&�%����׾������������ʾ׾�����������������������~�~�~���������������O�\�h�uƁƎƎƒƎƂƁ�u�o�h�\�X�P�O�L�O�Ϲܹ������������ܹ����������ùϺɺֺ�������
��������ֺ��������ɿm�y��������������y�m�`�_�U�R�T�V�`�b�m�`�m�x�y�|�z�y�m�i�f�`�`�`�`�`�`�`�`�`�`���'�3�@�Y�f�h�j�e�Y�L�@�3�'������������������������{�y�s�y������������������������	�������Ƨ�h�6�*�!�%�6ƁƜ���Y�e�r�~�������������~�r�e�Y�L�C�F�L�V�YÇÓàäìðôìàÛÓËÇ�z�y�y�zÀÇÇ���)�4�;�=�9�1�)�������������������)�6�B�O�[�h�zĀĀ�t�h�O�B�=�4�1�)�#�&�)���������������������g�A�(�!�$�/�A�Z�����;�H�T�a�d�m�n�r�m�g�a�T�H�;�7�/�*�$�/�;�����������������������������������~�~���������������������������������������������ѿ�����)�5�4�*�����ݿͿĿ���������
������������ؽнҽݽ�����������������������������y�q�n�o�y�����čĚĦĳĿ������ĿĳĦĚđčĄ�āāċč����0�<�I�P�W�V�Q�I�0������������������<�H�P�U�W�U�L�H�<�1�0�7�<�<�<�<�<�<�<�<�������ĿοͿĿ��������}�y�m�a�`�f�m�{���4�A�M�Z�a�f�j�m�h�Z�M�A�4�(�#� �"�(�*�4E�E�E�E�E�E�E�E�E�E�E�E�EuEtEkEmEuEvE�E��������ʼ̼ʼǼ�������r�f�d�r��������������(�*�(�#����������������M�Y�f�r���������r�f�Y�M�@�<�8�;�@�D�MŭŹ����������ŹŭŧŠŜŠšŭŭŭŭŭŭ���'�4�4�9�4�'������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DwDrDuD{D�D� Z P o E ^ 0 1 3 \ % 1 1 ( & J ' * M Q b > c ` ; J : & " l L 1 u $   H : C G H g % (   @ H   Y '   X     m  w  F  r  �  z    S  �    5  �  �  E    j  `      �  &  V  �  �  N  �  H  |  e  �  d  �  �  �  p    7  {  �    �  O    �  �  �  �  �  8  �  �  �  �  z  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  |  w  q  l  f  `  X  O  G  >  7  1  +  %        	    �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  _  \  Y  U  R  O  O  R  U  W  Z  \  Y  H  7  '       �  ^  c  f  b  ]  T  K  @  ;  A  :  %    �  �  �  �  �  �  b  E  J  P  V  [  _  ]  [  X  V  S  P  L  H  E  4    	   �   �  �  �    7  R  b  h  `  C    �  �  X  �  �     e  �  =  |  #  �  �  �    4  >  9  0  )    �  �  t    �  <  �    �  �    M  �  �  �  �  �  �  r  K  '    �  �  �  e    �  i  $  *  0  5  ;  A  G  E  ?  9  3  -  '  "  !             J  q  �  �  q  \  G  5  *      �  �  n  &  �  p  �  �   �  ]  h  r  s  m  f  Y  K  <  ,      �  �  �  �  �    _  ?  �  �  �  �  �  �  �  w  m  c  Y  O  ?  /  "      	     �  R  T  T  S  R  M  E  :  -      �  �  �  �    P    �  �    (  3  5  3  -      �  �  �  �  R    �  q  �    	   �  �  �  �  �  �  �  �  �  �  �  w  X  0  �  �  x  /  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  -  �  �  �  M    �  D  I  D  ;  '    �  �  �  �  s  k  g  {  p  O    �  �  =  �  �  �  �  �  �  �  �  �  �  ~  u  l  b  U  H  ;  /  "        �  �               �  �  �  �  �  t  O  "   �   �  ^  \  Z  W  T  M  G  @  :  4  -  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  @      �  �  �  �  �      �  �  �  �  z  I    �  �  *  �  �  �  0  �  }  �  �  ,  b  w  �  {  j  V  C  ,    �  �  a  �  =  V  {  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    z  v  r  m  h  �  �  �  �  �  �  �  h  U  ?  #  �  �  �  �  Y  &  �  �  �  "    �  �  �  �  �  [  *  �  �  _    �  H  �  |  �  \  %  y  �  �  	u  	�  
  
=  
d  
r  
q  
[  
/  	�  	t  �  -  ^  J  �  �  �  �  �    i  J  (  �  �  �  M  �  �    
^  	�  �  1  �  Y  �  �  �  �  �  �  �  �  �  �    t  i  ^  S  H  <  1  &    �  �  �  �  p  _  X  {  x  l  W  :    �  �  |  A  �  �  �  K  E  ?  9  3  -  &           �   �   �   �   �   �   �   �   �  �  �      !      �  �  �  R    �  �  J  �  p  �    �      
  �  �  �  �  �  �  Z  &  �  �  B  �  r  �  n    �  �  �  �  �  �  �  g  H  '    �  �  ~  K    �  U  �  (  �  #  �  �  �  m  �  V  �  �  �  T  �  �  �  m  �    !  �  �  �  �  �  �  �  �  z  @  �  �    �  �  	  V  �    �  0  I  +    �  �  r  [  N  7        �  �  �  �  �  `  �  �   �  �  �  �  �        �  �  �  {  F    �  �  9  �  �  t    {  ^  >  $    �  �  �  �  z  P    �  �  F  �  k  �  q  B  �  �  �  �  }  r  j  a  Y  Q  K  H  D  A  =  <  <  <  <  ;          �  �  �  �  �  �  c  P  8    �  t  �  �  �  <  p  R    !  (  4  >    {  p  ]  C  %    �  �  p  (  �  U  �  �  �    `  �  �  �  �  �  �  �  �  �  k    n  �  �   �  �  �  �  V    �  �  J  q  y  `  !  �  �  �  �  �  
m  �  p  �  �  �  �  �  �  �  o  H  &  �  �  �  ^    �  �  1  <  B  /  1  0  (         �  �  �  �  |  Z  2  	  �  �  n  ,  �  (    
     �  �  �  �  i  C    �  �  �  �  `  2    �  �  �  �  �  �  �  �  �  �  �  �  �  �  e  @    �  �  0  �  t  �    <  Y  i  a  Q  5  	  �  �  J  �  Y  
�  	�  w  Y  +  �    �  �  �  �  �  �  �  �  �  q  G    �  �  �  d  8    �  �  �  �  �  |  a  F  )  	  �  �    E    �  ;  �  2  �  '  �  �    .  4  2  &    �  �  �  b    �  j  �  X  �  ,  �  7      �  �  �  w  L    �  �  k  -  �  �  k  %  �  �  a  �  �  �  �  �  �  �  �  �  }  m  \  K  4    �  �  �  \    �  �  P  �  �  �  �  |    �    q  �  �  �    )  �  {  =