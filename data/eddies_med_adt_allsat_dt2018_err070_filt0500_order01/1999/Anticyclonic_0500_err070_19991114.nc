CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��E���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       P�{�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =Ƨ�      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @E��
=p�     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @ve�Q�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @M�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @��`          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ;ě�   max       >�p�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-_      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��-   max       B-0�      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��Y   max       C��      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�AA   max       C�{      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         X      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       P&��      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��{���n   max       ?�t�j~�      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��o   max       >?|�      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�G�z�   max       @E��
=p�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @ve�Q�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @M�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         G&   max         G&      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?蒣S&     �  QX      
   
         !               
         -                          W   	         1   
               	               -   C   &            �   1      3   $         *   @   8            O+�lO4�.O	�`O�rN���OJ�NB�0N��'M�D�N�^xNZ�NU<�NӜ�P��N�.Oj��N�{CO;uNM�M�Oa�aO���P�{�N��ANd�wN�O�?xN�,O�m�OK�[N�s�O<�BN{7�N��~N�OOV��Nu3�O�xlPE'O�N�M�N|s7OE4O�$RO�.�N�#�P ��OM�=N'L7N�_hO��cO���OZdN�TRNE�N_oO�ѵ�o��o;o;�o;�`B<#�
<D��<T��<�o<�C�<�t�<�t�<�t�<�t�<�1<�1<�j<���<���<���<�/<�/<�/<�h<�h<�<�<�=o=o=+=C�=\)=�P=�w='�='�=0 �=<j=@�=@�=L��=]/=q��=u=u=}�=�%=�C�=�O�=�t�=���=���=�{=�{=ě�=Ƨ�<<BHNR[gjtuyxtpg[IB<�����&)0.))!�������������������������� 
	�����������	 ��������
#/<DHPWRH</#
tt���������~xtttttttlklnuz}zz������zupnl�����������������������������������������x|����������������������������������������������  ����������������
�������EGHLUantvz~znmaWUHEE��������������������f`hhot����������thff������������������ �����������	

�����������������������)38@H?BMO5�� *BNg�������t[B5%��������������������#0<BDA<0&'#
)5:=53)qy����������������uq�������

�����,5INYYY^YNB-)��������������������������������������	
)6:BDFFB6)	ypnqz�����{zyyyyyyyy�������

�������XXU[ahtvtnhb][XXXXXXVYXabnz���������zmaV������������������������
/<QTOMH#
����������
)+#
�������s{����������������xs�������

���������������������������������������������
�����=9;@N[jx~��|ztg[NB=}�����������������������:FGC;)����{��������������������������������������� �����������������yz����������������y��������������TQUanz�������znla_XT��� ��������4*,5BCJIB54444444444�������������������������
	������b�n�r�{ŃŇōŉŇ�~�{�n�b�Z�U�J�L�U�[�b�����������������������������������������ܹ���������
������ܹܹԹӹܹܻx�����������������������x�k�_�Z�_�l�m�x���������������ټּӼԼּټ�����������������������������������M�N�Y�M�I�A�4�(�$�(�1�4�A�M�M�M�M�M�M�M����������������������������������#�/�0�<�=�<�6�/�,�&�#� �#�#�#�#�#�#�#�#�A�M�Z�^�f�s�t�y�s�f�Z�M�F�D�A�@�A�A�A�A�����ûǻĻû������������������������������û˻ͻȻû��������������������������������������������������x�u�z���������������(�A�M�j�r�v�x�s�f�Z�M������ݽսݽ�����������������������������������������˻��!�-�1�)�;�;�:�;�5�-�!����������-�:�D�F�S�[�_�`�_�S�F�B�:�8�-�*�&�+�-�-���
�
���#�)�&�#���
�����������������g�s�������������s�k�g�a�g�g�g�g�g�g�g�g�����������������������������������������*�6�C�O�\�^�^�O�6�4�*��������)�*�Z�f�k�l�u�u�z���s�Z�M�H�A�4�4�;�J�U�Z��)�6�C�G�F�<�)��������êäïÿ������������������������������z�|��������������������r�f�a�Z�f�r�|�����������������������������������������������������%�)�*�)�-�%����������������������;�H�P�T�]�a�b�a�a�Z�T�H�@�;�8�7�.�/�2�;�G�T�`�y�������y�w�m�`�T�G�;�6�5�;�=�E�G����������������������{�r�k�j�e�c�g�r��V�b�o�x�{�ǂǆ�{�o�n�b�Z�V�P�P�V�V�V�V���ʾ׾����ܾ׾ʾ��������������������!�-�:�F�N�Q�F�:�-�'�!� �!�!�!�!�!�!�!�!������	�	���	������ܾ��������'�1�4�8�4�4�2�'�!���
������ŔŠŭŷŹ������������ŹŭťŠŗŏŎŎŔ���� �����������������������������������m�y�������������y�m�`�G�B�2�7�?�G�S�`�m�5�N�g�t�~��g�N���������������5�M�Z�f�s�}�����}�f�Z�M�4�*�#� �"�(�.�4�M�Z�f�s�s�������y�s�f�Z�P�M�K�M�R�Y�Z�ZÓÞàëàÞÓÇÁ�ÇÈÓÓÓÓÓÓÓÓ�;�G�T�`�m�w�m�l�`�]�T�G�;�.�"���"�,�;D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DvDuD�D�D����5�A�O�T�N�A�5�����ݿٿ������ĿѿܿѿѿſĿ����������������������Ŀ�������������������������������������������������������������ùìäàÜÖÛàì���/�<�F�G�<�/�#��#�.�/�/�/�/�/�/�/�/�/�/������������������������������������������%�4�C�M�Y�`�]�M�4�'������������e�~�������z�r�_�L�@�'�����&�3�@�L�e�ܹ���������ܹϹù��������������ùϹ�Ƴ����������ƻƳƧƧƝƢƧƯƳƳƳƳƳƳ�uƁƎƓƏƎƁ�u�o�t�u�u�u�u�u�u�u�u�u�uE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������#�1�7�3�0�#��
������ľĶĿ����   / 8 F > C ] v v D 2 S R I 2 \ 1 ( [ r - R  0 N j 4 ? 5 d G H 3 < u L r 1 ! ( ! : 7 2 3 j . ' f & ; @ ; 8 E U l    n  �  0  >    �  �  �  V  �  %  �    �    ;  �  �  y  I  �  e    �  �      )  4  
    �  �  �  �  �  �  �  B  �    �  �  �  �     [  �  ,  �  �    �  �  E  �  �<o;ě�<49X<��
<��=0 �<u<�1<��
<�/<�`B<���<�=�%<�/=,1=C�=0 �<�<�/=��=49X>�p�=��=C�=#�
=��w=#�
=L��=T��=8Q�=@�=0 �=0 �=<j=y�#=49X=�-=�l�=�1=�o=�C�=���>E��=�;d=�C�=�l�=���=���=��T=�x�>\)>V=�j=�^5=�/>   B��B3�B ��B�OBٷB[�B�BK�B�OBG_BAB ��B"��B#HBA�B �-BitB��B�7B�$B�BW�B�jB!,�B%��B\�B�JB;�BM�B �yB�8B9�B_�B#˕B�~A���B��B��B݇B�(B
B"bEBq�B��B�CB
��BԑBY�B�/B-_B�%BywB�Bt3BPB۞B��B�6B HB ��B�uBõB�-BЙB?uB�xBF5B��B ��B"�B#@�B;fB �BC�B��B��B�(BʕB}�B��B!@{B%ڟB��B�BG�B@AB ʸB�bB?�BW�B#�yBD�A��-B>�B�=B9cB�GB@,B"@�B�]B��B��B
��B�$BE�B�B-0�B<�B}�B?qBM�B�uBA�B��A�GA���?)QE@�D�AqAҵlA:�!A�\�A��A>�^@�<@�3@��A7߽A�Fs@`�@�j�A�@�A��A��CB TSA?I�A��A+x@�A��A���A�zAha�@�phB��AP+�@z�`AX,/@ǂ/A��vA���Aj�NA�K�A=V\AA8A�dIAd˘C���A��QAv�A�%EA͓�A���A!��@Ǌ�?�v�>��YBO�BLC��A�+�A�oaA�~J??�@�LA�xAҀA;^�A���A"A?�@��@��}@�VA7�WA���@\6�@���A��xA�A��^B wA>�AҡlA�2@�Z�A��dA��-A��pAh��@� B�xAP�@ym)AW��@��A��A��Ai�?A�~{A<��AAAʀ�Ad�C���A��EAw)A��À�AuA!��@�Q?��Y>�AABz�B4FC�{A斓         
         "      	                  .                          X   
         1                  	               -   D   &            �   2   	   4   %         +   A   9                                                      '      !                  !   3            #                                    /   !               !      %            #   %                                                         !                                       !                                    +                        !            #                  NO4�.O	�`O�rN���OG�NB�0N�F�M�D�N�^xNZ�N��N��O�\aN�O�O/��N�{CO�NM�M�Oa�aN���O���N��ANd�wN�O��N�,Oz6�N���N�s�O<�BN{7�N@-�N�OO:Nu3�O��P&��O��[N�M�N|s7OE4O[O�&�N�#�O͎�O�zN'L7N�_hO��cO���OZdN�TRNE�N_oO�ѵ  
  d  @    ~  �  K  �    )  7  ]  �  L  �  �  �    7  ~  	  o     =  [  b  <  �  �  +  �    �  �  X  �  �  m  Q  �  �  L  �    �    �  �  8  �  �  	X  �  r  �  �  �o��o;o;�o;�`B<�o<D��<e`B<�o<�C�<�t�<��
<�1<�h<�j<ě�<�j<�/<���<���<�/=\)>?|�<�h<�h<�=+<�=C�='�=+=C�=\)=��=�w=0 �='�=e`B=T��=e`B=@�=L��=]/=���=�o=u=�O�=�O�=�C�=�O�=�t�=� �=���=�{=�{=ě�=Ƨ�DFMNW[bgqttutkg[TNDD�����&)0.))!�������������������������� 
	�����������	 ��������#/<?HLSNH</.#tt���������~xtttttttmlmntz~�����zvqnmmmm�����������������������������������������x|���������������������������������������������������������������

��������QLPUamnrwnha\UQQQQQQ��������������������f`hhot����������thff���������	������������ �����������	

��������������������	)55952)14:BN[gt|���tg[NB71��������������������#0<BDA<0&'#
)5:=53)v{���������������wtv�������

�����)5BNVWW[UNB2)����������������������������������������	
)6:BDFFB6)	ypnqz�����{zyyyyyyyy�����

���������XXU[ahtvtnhb][XXXXXX][[aempz��������zma]�����������������������
/<EGE</#
��������
 ' 
�������~������������������������

������������������������������������������� 


������?:;=BN[gt|~~ytg[NB?}���������������������)6AC>6)��������������������������������������������� �����������������yz����������������y�������		��������TQUanz�������znla_XT��� ��������4*,5BCJIB54444444444�������������������������
	������b�n�{�|ŇŊŇŅ�{�q�n�b�`�U�P�R�U�Y�b�b�����������������������������������������ܹ���������
������ܹܹԹӹܹܻx�����������������������x�k�_�Z�_�l�m�x���������������ټּӼԼּټ�����������������������������������M�N�Y�M�I�A�4�(�$�(�1�4�A�M�M�M�M�M�M�M�������� �����������������������������#�/�0�<�=�<�6�/�,�&�#� �#�#�#�#�#�#�#�#�A�M�Z�^�f�s�t�y�s�f�Z�M�F�D�A�@�A�A�A�A�����ûǻĻû������������������������������ûȻ˻Żû�������������������������������������������������������������������4�A�S�]�f�j�c�Z�M�A�(��	���������4���������������������������������������˻��!�&�-�8�8�/�-�*�!������������-�:�D�F�S�[�_�`�_�S�F�B�:�8�-�*�&�+�-�-�����
��� �#�&�#�!��
�����������������g�s�������������s�k�g�a�g�g�g�g�g�g�g�g�����������������������������������������*�6�C�O�\�^�^�O�6�4�*��������)�*�Z�f�m�l�m�f�e�Z�S�M�E�H�M�N�Z�Z�Z�Z�Z�Z����$�*�+�*�%����������������������������������������������z�|��������������������r�f�a�Z�f�r�|�������������������������������������������������	���#�(�(�&�)���������������������	�;�H�P�T�]�a�b�a�a�Z�T�H�@�;�8�7�.�/�2�;�T�`�m�y�����}�y�s�m�`�T�G�;�8�7�;�?�G�T���������������|�r�r�m�r�t�������V�b�o�x�{�ǂǆ�{�o�n�b�Z�V�P�P�V�V�V�V���ʾ׾����ܾ׾ʾ��������������������!�-�:�F�N�Q�F�:�-�'�!� �!�!�!�!�!�!�!�!�����	��	������������������'�1�4�8�4�4�2�'�!���
������ŔŠŭųŹ������������ŹŭŧŠŘőŐŒŔ���� �����������������������������������m�y�������������y�`�T�I�B�<�B�G�M�T�`�m�5�B�g�q�t�q�s�p�g�N���������������5�A�M�Z�d�t�y�x�t�f�Z�M�A�4�2�*�&�*�/�4�A�Z�f�s�s�������y�s�f�Z�P�M�K�M�R�Y�Z�ZÓÞàëàÞÓÇÁ�ÇÈÓÓÓÓÓÓÓÓ�;�G�T�`�m�w�m�l�`�]�T�G�;�.�"���"�,�;D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����(�5�A�M�R�M�D�5�(�����������ĿѿܿѿѿſĿ����������������������Ŀ�����������������������������������������àìù����������������ùìéàßÙßàà�/�<�F�G�<�/�#��#�.�/�/�/�/�/�/�/�/�/�/������������������������������������������%�4�C�M�Y�`�]�M�4�'������������e�~���������~�r�e�Y�L�3�'����,�@�L�e�ܹ���������ܹϹù��������������ùϹ�Ƴ����������ƻƳƧƧƝƢƧƯƳƳƳƳƳƳ�uƁƎƓƏƎƁ�u�o�t�u�u�u�u�u�u�u�u�u�uE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������#�1�7�3�0�#��
������ľĶĿ����  / 8 F > 3 ] d v D 2 R ? ? , O 1 & [ r - A  0 N j 4 ? 0 S G H 3 4 u H r *  " ! : 7   ' j &  f & ; 8 ; 8 E U l      �  0  >    H  �  �  V  �  %  6  �  �  �  �  �  J  y  I  �  �  �  �  �    �  )  �  �    �  �  X  �  �  �  %  �  )    �  �    i     �  F  ,  �  �  �  �  �  E  �  �  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  G&  �  �      
  	    �  �  �  �  p  J  .    �  �  !  ~   �  d  b  `  ]  [  W  P  C  5  %      �  �  �  �  �  i  =    @  3  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        	  �  �  �  �  �    Y  =  ?  4    �    �  �  V  ~  t  i  ]  M  6    �  �  �  t  4  �  �  q  Q  %  �  v    o  �  �  �  �  �  �  �  ^  /  �  �  o  .  �  Q  �     w  �  K  P  U  Y  ^  c  h  f  a  \  W  R  M  G  @  :  3  -  &    �  �  �  �  �  �  �  �  �  m  M  &  �  �  �  {  N  :  N  c        �  �  �  �  �  �  �  �  �  �  �  �  V  &  �  �  �  )      �  �  �  �  �  �  f  K  1    �  �  i  (  �  �  a  7  g  �  �  �  m  W  @  &  
  �  �  �  y  S  -    �  �  �  @  F  L  R  W  [  ]  \  \  H  1                    w  s  v  �  �  �  �  �  �  z  l  \  G  0    �  �  �  j      &  1  :  H  K  B  2       �  �  �  r  6  �  �  �  %  K  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  O  #  �  �  �  �  �  �  \  
  �  �  y  p  j  g  f  e  c  _  Z  U  H  8    �  �  N  �  �  G          �  �  �  �  �  a  2    �  �  �  �  �  �  �  �  7  .  $        �  �  �  �  �  �  �  �  �  �  �  l  V  ?  ~  x  r  l  f  a  [  U  O  J  3    �  �  �  m  F     �   �  	    �  �  �  �  �  �  �  �    q  s  j  ^  J  *  �  �  6  �  �  �      /  K  e  l  n  k  c  S  ?  &  �  �  D  �  y  %  �    �  2  a  6  �  �  �  �  >  X  4  �  �    �  Z  �  =  3  *      �  �  �  �  �  �  f  H  +    �  �  �  z  C  [  W  R  N  H  =  2  '        �  �  �  �  �  �  j  F  !  b  L  6      �  �  �  �  k  [  Z  B    �  �  �  f  5    2  <  7  *        �  �  �  �  \    �  j  �  s  �  �  �  �  �  �  j  U  D  0    �  �  �  �  i  G  '    �  �  �  y  �  �  �  �  �  �  �  �  q  V  7    �  �  �  a  0    �  �  r  �  �  �  �  �    !  )  (  "       �  �  N    �  &  `  �  �  �  �  �  �  �  �  �  �  �  �  c  C  #    �  �  �  �            �  �  �  �  �  �  �  q  P  &  �  �  �  ?   �  �  �  �  �  �  �  �  �  �  �  �  y  m  `  S  ;     �  `  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  a  h  n  t  X  H  8  )      �  �  �  �  v  \  @  $    �  �  �  �  r  �  �  �  �  �  �  m  L  %  �  �  �  i  (  �  �  K  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �       4  N  �  �     =  Z  j  m  h  [  C    �  w    �  =  �    '  �  �  9  O  A  4  /  (    �  �  �  W    �  C  �  '  �  �  \  p  �  �  �  �  �  �  �  k  H    �  �  x  *  �  w    |  U  �  �  �  �  ~  f  M  4       �  �  �  G  �  �  [    �  S  L  9      �  �  �  h  :  	  �  �  V    �  R  �  �  /  �  �  �  �  �  �  �  �  �  o  O  $  �  �  �  U    �  �  3  �  �  V  �  �  .  �  �  �        �  ^  �  �  �  i     3  �  U  �  �  x  \  9    �  �  ~  1  �  l     �    �  L  �  M      �  �  �  �  �  �  �  ~  _  ?       �  �  �  k  �  /  �  �  �  �  �  �  p  T  9    �  �  S  �  �  1  �  &  �  �  �  �  |  �  m  I     �  �  o  !  �  w    �  U  �  j  �  �  8  *      �  �  �  �  �  �    _    �  �  Q     �   �   o  �  �  �  �  �  s  [  B  (    �  �  �  �  \  4  	  �  �  m  �  �  �  �  }  x  p  g  \  Q  B    �  �  8  �  O  �  �  +  �  	4  	P  	W  	L  	4  	  �  �  b    �  �     �  
      z    �  �  z  Q  "  �  �  Z     
�  
2  	�  	  h  �  �  �  �  G  �  r  h  ^  T  I  >  3  )      �  �  �  �  �  r  Q  .    �  �  w  h  Y  H  6  $      �  �  �  �  �  �  �  �  {  B  	  �    �  �  �  y  J    �  �    H    �  �  �  ^  7  	  �  �  �  �  �  �  �  d  E  #  �  �  D  �  �  -  �    w  �  "