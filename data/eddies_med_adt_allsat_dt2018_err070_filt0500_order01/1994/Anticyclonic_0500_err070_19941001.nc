CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��\(�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�u#   max       P��+        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       =ȴ9        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F�\(�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vip��
>     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @Q            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @�             5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;��
   max       >��;        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��<   max       B3��        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��F   max       B3�&        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�:�   max       C�e        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��l   max       C��        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          K        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�u#   max       P��+        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����o   max       ?�z����        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >M��        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @F�\(�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vip��
>     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P`           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @�             U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?�z����        W�               
   (         "                  %        �      �            *   +   	                        D            )                  M   +   
            	       	   	   
   	         #   b         Q      ,N�*_NZ�OQZ�N��[N��CP�
3N�g�N"H�OMn�N� N��N��OP�}N�mO��O�{�O`TP��O�p�O��NL�N��N�c^P��+O��Ow�O��IOȰN��N�O[O��N|�Ng;XP�)N�!�N�pN��.Oթ�OÖ
N�o�N��fN*�M�u#PK!9O��N���OM�6N=k�O��Nf�GO3��N�N.$�N�mdN@�Oq2�O$ޫO��O��TO�'OT��O��
N8(�OUƉ�D��%�  :�o;D��;�o;��
;��
;��
;��
;�`B;�`B<t�<#�
<49X<D��<D��<u<�o<�o<���<���<��
<��
<�1<�1<�j<�j<�j<ě�<�`B<�h<�h<��<��<��=+=�P=�P=#�
=#�
=#�
=#�
=#�
=0 �=0 �=49X=49X=49X=8Q�=H�9=Y�=Y�=m�h=q��=q��=u=�%=��=�7L=��w=�{=�9X=�9X=ȴ9��������������������9657<HIKKKI<99999999837<HUanplnu{qnaUH<8KHMOW[hjqkh][OKKKKKKbehnt�����thbbbbbbbbXWz�����������zimfeXZ[abmz|zyz~zmaZZZZZZfegtt�tmgffffffffff��������������������""&/9;AB;;/*$""""""" "/30/&"          //:<HHMKH<://///////��������������������)*55.))BN_gt~}yt[N;32.%������

��������������������������'#,A[g���������gNB1'96HNgt������ng`[UJC9��������
 !
�����fa_`hntv{xthffffffffEEGN[ggigb[NEEEEEEEECBIUbenonib^UJKICCCC�����6HamHD/�������������������������� ��	)55@=5)) ������
#/<A?/
������������
 */#
����������������������|z����������||||||||Zakz����������znmcaZ��������������������������������������������������������������������������������



#0040(#




!%{uvy���������������{)6O_hmmppmh[OB2mpqtz����������tmmmm������

������

#/'#
2<>IU[WUI<2222222222���������$53)�������47.6DOLE:)��&)3)����34:AN[bgjmnmjg[NHB93/+/;<HJPHB</////////�����������������������������������3,*-36?BKOW[^c[OLB63XYY[^hpnhd^[XXXXXXXX*//<HRSH</**********����������������������� ����������������������������ltvz�������������zml����5DHHEB5)�{�����������������z{xtqnnz�����������~zx�������� ���������)BLB61)��ZOU[^hnrjhg[ZZZZZZZZ���������

����#�/�3�<�F�@�<�2�/�(�#����#�#�#�#�#�#�l�x�������������x�l�e�`�l�l�l�l�l�l�l�l�����!�"� �� �����������������������
���#�&�$�#��
��������������������@�L�O�Y�]�^�Y�L�@�5�9�;�@�@�@�@�@�@�@�@�����,��!��	���������g�W�G�J�Z���������;�H�S�T�\�W�T�I�H�G�;�1�5�9�;�;�;�;�;�;�����������������������������������������������������������������x�n�l�o�{���/�6�;�>�;�:�/�"�����"�)�/�/�/�/�/�/�H�T�a�d�d�a�T�Q�H�>�H�H�H�H�H�H�H�H�H�H�"�"�.�/�.�.�"�����"�"�"�"�"�"�"�"�"����'�,�1�3�/�'���
������������Z�f�q�s�v�t�s�f�Z�M�K�H�M�T�Z�Z�Z�Z�Z�Z�p�o�����z�s�a�H�/������"�/�;�H�e�p���(�4�A�M�q�f�Z�M�A�4�(��������àìôù������ùìÖÓÎÇÆÄÇÔÜÞà��)�B�[�n�q�m�f�O�6�����������������������������������������s�Z�K�A�1�A�N�s��DoD�D�D�D�D�D�D�D�D�D�D�D�D�DjD\D[DWDbDo�S�_�l�x��x�p�l�_�S�Q�P�S�S�S�S�S�S�S�S�)�5�B�K�H�B�B�5�2�)�!��)�)�)�)�)�)�)�)�����������������������������������������;�G�a�m���j�;�&�������������������"�;�������������������������������������������������������������������������������ž��.�A�M�Z�l�i�S�B�"�������������(�5�A�L�O�F�A�5�+������ �����s�~�������������������s�f�Z�]�f�m�s�s��������������ܾھ��������ĚĜĦĬīĦĦĚĐčċāĀ�z�yāĂčĕĚ�/�<�>�<�6�/�#����#�.�/�/�/�/�/�/�/�/������������������������������������������������������ìÓ�z�r�j�i�n�zÓàù���뿒�����������������������}���������������ּؼ������������߼ּӼӼԼּּּ����������������������������������������彷�нݽ��������ݽĽ����������|�{�������׾���������������׾������������ʾ׿	��"�)�.�5�7�.�,�"���	����	�	�	�	�����������ĺǺ����������~�z�|�~���������ֺ��������ֺպ˺ֺֺֺֺֺֺֺֺ��������������������������ƚƧƳ������������������ƧƇ�u�c�gƀƋƚ�Y�e�~�����ͺʺֺ�����ﺰ�����~�m�h�Z�Y�#�0�4�<�F�I�I�I�<�7�0�-�#����#�#�#�#�I�U�b�n�v�w�n�g�b�U�I�<�0�-�.�/�0�3�<�I����������������������������������������"�.�;�G�T�T�X�T�R�G�;�.�"������"�"��������������������~�{��������������������ûлܻ�����������ܻϻ������������'�0�4�;�4�'������������²¿������¿²°­¬²²²²²²²²²²���������������ݿۿտѿϿѿҿݿ���-�:�F�S�_�S�F�:�4�-�*�"�-�-�-�-�-�-�-�-�����������������������y�h�k�`�U�`�l�y��ŭŹ������������������������ŽŴŭŤũŭ���������	�����������¿²¬¥²�˹��������
��Ϲ������������ùϹܹ�_�l�x�����������������x�l�_�S�F�E�S�Y�_�������	��!�"�$�#��	�����������������伋���������ļǼƼ¼�������������t�q�v������������������������������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EwEuElEu : ^ 5 G ? I @ V  / C F 4 6 - S K  ? $ F + 5 S  Y H B F = ; j � < * P : N 0 5 K } � P d D  H 6 > M y n C i  k F 1 Q & [ c @  �  l  �  �  �  �  �  5  �  �  ,  I  �  �  8  e  f  *  Q  %  _  �  �  �  m  o  �    �  �  h  #  �  �  �  �  �    �  	    r  y  �  �  �  �  Y  F  j  �  _  �  *  t  �  �  �  $  f  �  �  ?  �<o;��
<�1<t�<T��=49X<o<o=�w<t�<t�<T��=o<�9X=H�9=��=,1>��;=49X>@�<�<�/<���=}�=�%<��=0 �=,1=��=+=,1=C�=8Q�=ȴ9=�P=��=L��=��-=�O�=P�`=Y�=49X=0 �=�F=�{=Y�=�%=P�`=��=ix�=�1=y�#=�+=�O�=�C�=���=��T=���>(��=ě�=\>,1=��>hsB MB&m�B��BBgBLBhA�]�B	�3B"��A�́A��<B��B QxBW�B\UB#}LB" �B	XB	\yBB�B��B'M�B��B��BB�B7UB/'B��B3��A���B� BA9B�Bt>B$�)B�*B�RB^B��B#��B$�B&��B�rB�.B��Bu
BʝBӂB#M'B_�B�FB�)B��B�,B,t_B �gBY8B��B�QB��B�VB��B�B ;�B%ЇBH�B<�BIhB�cA��B	�OB"��A��4A��FB-B�iB�GB?�B#�B"@�B	?�B	��B;3B�B�]B'�hBE�B��B��B?�BC2B��B3�&B =.B�aB@B@ B?B%45B��B��B��B.�B#��B$DkB&��B��B�nB�B�4B��B�B#?OBB�B��B��B�/B�B,�*B?�B�|B��B��B�BI�B�B?�A�h�@��CA�.gA�ژ?�u�A���A�D\A�\~@��-A��A��RA^��?y�zA@�A�yJA9�LA��xA�J�A��HC�ѭ@��5A�y�@�JA�r�A�aqA�|nA7��A�'+ADuAV�A�
&A�jA�rA̒�Aq�EAI�A�``A%�BAR&�A]�	@�4@A��@MiB�M@HoA�a$A�kA�HiAas�@�k@���@ǯfA���A~��@�A��A�OjA�ہ>�:�@��A�w�@���@�يC�eA�~�@���Aғ�A��a?�jA���A��A���@�@�A��|A��hA^�K?n�HA?	�A�}�A:E�A�M�Aԧ9A�n�C�מ@���A��@��A��UA��A��wA7lA�s�ADLAW
A޴�A��LA�~�Å�Ar�A aA�~�A&�AQ	�A^�!@�)@<-@LatB3�@'gA�LA�CA��WAa T@��@��@�%�A�G\A~�f@|QA�A�wA�hK>��l@��.A��s@�ۮ@�2�C��               
   (         #                  &        �      �            *   +   	                        D            )                  M   +   
            	       	   	      
         $   c         Q      -                  ;                           #         3   %   !            K         '                     )            #   !               1   /                                          %         !                                                               #               K                                                               !                                                         NO��NZ�N�>NVl�N��CO�9�N�g�N"H�N�}�N� N��N��O1�1Nl��N�6FOp��NճOǤ�O���O8_�NL�N��N�A�P��+O���Ow�O9ފO'�8N��N�O[OcN|�NM$6O�m�N�!�N�pN��.O<�O��=N�o�N��fN*�M�u#O�1HO���N��
O<�HN=k�N�iNf�GO3��N�N.$�N�mdN@�N�VN��qO��gO���O�'OT��Oz�?N8(�OUƉ  %  �  Y  d  �  *  �  �  �  W  �  �  y  �  �  �  [   R  9  �  z    +  �  h  7  }  �  f  �  i  +  �  	f  '  B  �  A    9  �  �  @  	5  �  @  h  �    �    �  -  �  �  �  i  �  0    	  ]  p  ��o%�  ;ě�;�o;�o<�`B;��
;��
<�t�;�`B;�`B<t�<D��<e`B=o<u<�t�>M��<�t�=�v�<���<��
<�1<�1<�`B<�j<�<�`B<ě�<�`B<�<�h=o=@�<��=+=�P=T��=0 �=#�
=#�
=#�
=#�
=�\)=D��=8Q�=8Q�=49X=@�=H�9=Y�=Y�=m�h=q��=q��=��=�+=�7L=ȴ9=��w=�{=��`=�9X=ȴ9��������������������9657<HIKKKI<99999999;87<HQUahnfaUHF<;;;;NJOPY[ghphg[QONNNNNNbehnt�����thbbbbbbbb��������������������Z[abmz|zyz~zmaZZZZZZfegtt�tmgffffffffff��������������������""&/9;AB;;/*$""""""" "/30/&"          //:<HHMKH<://///////��������������������)02)=9>BHNX[eghge[[NIB==�������

�������������������������@?AN[gt�������tg[NC@<:M^gt������wkg[MHF<��������

�����fa_`hntv{xthffffffffEEGN[ggigb[NEEEEEEEEDDIU`bmgbZUPMIDDDDDD�����6HamHD/�������������������������� ��	)55@=5)) ����
#)/25/#
��������

�������������������������|z����������||||||||`ammz��������zpmea``��������������������������������������������������������������������������������



#0040(#




!%������������������)6BO[hmnmkh[OB4!mpqtz����������tmmmm������

������

#/'#
2<>IU[WUI<2222222222����������
	����������)6BHJHB6)�� $)-)      =66;BCN[bgjlmlig[NB=/+/;<HJPHB</////////�����������������������������������3,*-36?BKOW[^c[OLB63XYY[^hpnhd^[XXXXXXXX*//<HRSH</**********����������������������� ����������������������������{|����������������{{����5BGGDB5)���������������������xtqnnz�����������~zx�������� ���������$)& ����ZOU[^hnrjhg[ZZZZZZZZ���������

����#�/�<�C�>�<�0�/�.�#��!�#�#�#�#�#�#�#�#�l�x�������������x�l�e�`�l�l�l�l�l�l�l�l���������
��������������������������
���#�%�#�"��
�
�������������������@�L�O�Y�]�^�Y�L�@�5�9�;�@�@�@�@�@�@�@�@�����������������������������������������;�H�S�T�\�W�T�I�H�G�;�1�5�9�;�;�;�;�;�;�������������������������������������������������������������x�}�������������/�6�;�>�;�:�/�"�����"�)�/�/�/�/�/�/�H�T�a�d�d�a�T�Q�H�>�H�H�H�H�H�H�H�H�H�H�"�"�.�/�.�.�"�����"�"�"�"�"�"�"�"�"����"�'�+�/�1�-�'��������������Z�f�k�p�n�f�Z�P�M�M�M�Y�Z�Z�Z�Z�Z�Z�Z�Z�H�T�a�g�m�r�m�l�a�Z�T�O�H�;�;�6�;�A�H�H��(�4�A�H�M�b�f�m�f�Z�M�A�4� �����àìðù����ùíìàÓÈÇÆÇÓ×ßàà���)�?�I�N�L�B�7�)���������������������������������������s�g�Z�R�A�N�g�s��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDvD{D�D��S�_�l�x��x�p�l�_�S�Q�P�S�S�S�S�S�S�S�S�)�5�B�K�H�B�B�5�2�)�!��)�)�)�)�)�)�)�)�����������������������������������������;�G�a�m���j�;�&�������������������"�;�������������������������������������������������������������������������������ž�(�4�9�A�M�P�Q�I�A�4�(����������(�5�A�G�J�A�=�5�(������������s�~�������������������s�f�Z�]�f�m�s�s��������������ܾھ��������ĚĚĦĦīīĦĥĚčāā�{�zāĄčęĚĚ�/�<�>�<�6�/�#����#�.�/�/�/�/�/�/�/�/����������������������������������������ù����������������ìàÓÇ�}�t�t�yÓàù�������������������������}���������������ּؼ������������߼ּӼӼԼּּּ�����������������������������������������Ľнݽ���ݽڽнĽ������������������ľ׾��������������׾ʾ������������ʾ׿	��"�)�.�5�7�.�,�"���	����	�	�	�	�����������ĺǺ����������~�z�|�~���������ֺ��������ֺպ˺ֺֺֺֺֺֺֺֺ��������������������������Ƴ��������������������ƳƤƝƙƙƚƞƧƳ�r�~�����źƺκкպɺ������{�o�j�b�^�a�r�#�0�2�<�D�G�<�3�0�/�#�"���#�#�#�#�#�#�<�I�U�b�m�n�u�v�n�f�b�U�I�<�0�.�/�0�7�<����������������������������������������"�.�;�G�K�T�O�G�;�.�"������"�"�"�"��������������������~�{��������������������ûлܻ�����������ܻϻ������������'�0�4�;�4�'������������²¿������¿²°­¬²²²²²²²²²²���������������ݿۿտѿϿѿҿݿ���-�:�F�S�_�S�F�:�4�-�*�"�-�-�-�-�-�-�-�-�����������������������y�y�s�y����������Ź����������������������������ŹŷŲŹŹ���������������������¿²®§²�˹Ϲܹ����������ܹϹù����������ùϻ_�l�x�����������������x�l�_�S�F�E�S�Y�_�������	��!�"�$�#��	�����������������优�����������üü������������}�x�|����������������������������������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EwEuElEu 9 ^ L M ? 4 @ V  / C F 1 3  M B 	 =  F + : S  Y * ! F = 8 j H 7 * P : C 2 5 K } �  J B  H 0 > M y n C i , W D " Q & 4 c @  o  l      �    �  5  �  �  ,  I  �  {  �    �  �  �  |  _  �  �  �    o  �  _  �  �  >  #  v  �  �  �  �  �  i  	    r  y  e  �  �  �  Y    j  �  _  �  *  t  �    �  
  f  �  �  ?  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�          &  *    
  �  �  �  k  /  �  �  s  0  �  �  Q  �  y  n  c  W  H  :  +    �  �  �  �  j  M  0     �   �   �  �     @  O  W  X  T  N  H  :  )      �  �  f  0  
  �  �  I  Q  Z  b  e  f  g  c  \  V  L  ?  2  %    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c    �  �  B  �  �  G  ^  f  g  _  _  �  �  �  �    '      �  �    �  �  )  �  �  �  �  �  �  �  �  �  �  �  }  y  u  n  h  b  \  U  O  �  }  u  n  g  `  X  U  T  S  S  R  Q  N  H  B  <  5  /  )  �  .  [  �  �  �  �  �  �  �  �  �  p  >  �  �  �  �    �  W  P  H  A  :  2  +  $           �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  l  f  `  Z  W  X  Y  Z  Z  b  m  w  �  �  X  x  u  d  P  2    �  �  �  �  �  �  �  �  q  J  �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  5  h  �  �  �  �  �  �  �  �  �  �  �  �  W    �  J  �  M  �  �  �  �  �  �  r  W  :    �  �  �  |  G    �  �  �  d    @  Y  P  6      �  �  �  f  8  �  �  3  �  v    �  M  �  0    �    S  ?  �   5   R   2  �      �  �  �  <  �  �  )  5  4  (    �  �  �  �  �  �  �  �  �  �  \    �  �   �  �  �  �  �  y    a  �  �  �  m  "  �  �  �  �  �  !  �  �  z  m  _  M  9  $    �  �  �  p  3  �      �  z  0   �   �          �  �  �  �  �  �  �  �  �  m  Z  J  :    �  �      !  )  *  (  '  !        �  �  �  �  �  v  a  L  8  �  �  �  �  q  F    �  �  �  �  �  i  V  ?    �  �  L  �  4  M  _  g  d  V  8  !       �  �  �  y  C  �  �  $  �  u  7  6  5  /  '      �  �  �  �  �  �  p  M  D  C  2    �  �      &  :  k  y  |  v  l  \  ?    �  �  y  3  �  �  j  c  �  �  �  �  �  �  �  �  k  J  &  �  �  �  J  �  �  R  �  f  ]  Q  @  *      �  �  �  �  �  _  0  �  �  �  `  %  �  �  �  �  �  �  �  �  x  o  d  Z  O  A  1  "     �   �   �   �  `  f  g  b  Y  N  C  5  $    �  �  �  W    �  ^    �  �  +    �  �  �  �  �  �  �  y  g  V  C  0    	  �  �  �  �  �  d  �  �  l  N  -    �  �  j  *  �  �  a    �  �  @  �  8  �  	  	G  	b  	a  	D  	  �  �  h  %  �  Y  �  "  W  d  B  X  '  $  "                �  �  �  �  �  x  T  )   �   �  B  >  :  7  2  *  "        
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  Y  6    �  �  �  R  �  f  �  �  �  �  �       0  @  <  4  &    �  �  �  [    �    H  �            �  �  �  �  �  �  v  G    �  e    �  �  9  6  2  +  "    	  �  �  �  �  �  t  U  6    �  �  �  c  �  �  �  �  �  �  �  h  D    �  �  �  �  c  C  )    �  �  �  �  �  �  �  {  ]  ?       �  �  �  �  j  J  )     �   �  @  >  ;  8  6  3  0  #    �  �  �  �  �  �  �  j  Q  9  !  '  �  �  �  �  	  	+  	4  	$  	
  �  �  �  9  �  E  �  �  �  �  �  H    ~  s  X  *  �  �  e  ,  �  �  �  I    �  O  �  J  3  9  ?  =  ;  7  4  3  4  1  +  !    �  �  �  �  ^    �  e  h  g  ]  Q  E  7  '      �  �  �  p  =    �  �  N    �  �  �  �  �  �  o  S  7    �  �  �  �  a  @    �  �  �               �  �  �  �  {  G    �  B  �  T  �  t    �  �  �  �  �  �  �  �  �  p  \  D  .        �  �  �  �      �  �  �  �  �  �  �  �  �  y  �  ~  %  �  .  �    {  �  �  �  n  ]  T  g  y  �  �  �  �  �  �  �  �  �  �  �  �  -  C  X  i  w  �  �  �  �  �  �  u  i  [  N  A  4    �  �  �  }  s  f  Y  M  ?  /      �  �  �  �  [    �  z     �  �  |  l  R  8    �  �  �  �  s  J     �  �  �  m  >     �  J  9  /  +  .  I  [  o  �  �  �  �  �    _  6  �  �  ;  �  -  9  <  b  \  E  "  �  �  �  g  D    �  �  �  R  �  L  �  �  �  �  �  �  |  r  `  >    �  �  b    �  p  �  u  �  ,  �  �      &  -  0  *    �  }    
�  
,  	o  �  �  )  �   d    �  �  �  �  f  =    �  �  �  u  M  %  �  �  �  �  q  I  	  �  �  �  �  �  �  �  v  Y  ;    �  �  �  �  �  �  o  M  �  �  ;  W  S  ?  "  �  �  I  �  R  
�  
  	[  �  �  L     �  p  T  7    �  �  �  �  �  k  M  /    �  �  �  �  �  u  _    
�  
�  
�  
L  
  	�  	q  	  �  4  �  8  �  %  �  �    R  q