CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��t�k      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =�{      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @D�z�G�     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ҏ\(��    max       @vq\(�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @R�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��1   max       =��      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��R   max       B&�D      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�t+   max       B&��      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�O   max       C�p�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?N��   max       C�j�      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          V      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       Pe�b      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��2�W��   max       ?�}Vl�!      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       =�{      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @D��Q�     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ҏ\(��    max       @vq\(�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @R�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��)^�	   max       ?�}Vl�!     p  S(            P         	         
             	                  	   
   /   A      A            (         '         -            #      !      V      !               (   (   A      2               N��N�N�BbP���N�@N���N���N��N\c�Nt�NmRVOOI�NW2N�d>NM�%O%8NԍN�x-OR�O ��O��P
3�P��N�B�O�+O�[�O��>N76O���N�ߖN�B/P7O�O�yO�M�nkO��#O�O�P�O���N�X�P%��O)]�O���N�>�M��NL��N2��O��'O~eO��bOIz9O���N�tgOTL�N�I�O�N���j��C��o�ě��ě���o�D��:�o;o;��
;�`B<o<o<t�<#�
<D��<D��<T��<u<�o<�C�<�t�<�t�<�t�<��
<��
<��
<�1<�9X<�j<�j<�/<�/<�/<�/<�=+=+=C�=\)=��=��=�w=�w=#�
=,1=0 �=49X=8Q�=H�9=ix�=q��=}�=�hs=�hs=��w=���=��=�{~���������������~~~~��������������������;<9::<DIUUZXVUJIE<;;���.BNgux[OB5����������������������������������������������������������������@?BBOTTOKB@@@@@@@@@@��������������������������������������fdcfmoz{}{zmffffffff�������������������������������������������������������������������������

�����!%)58<5-)!!!!!!!!!!05679@BOP[XROGB6000040-)6BEKOW[aa[OB=864!')+268?B<>)t�����������������t� 	/;HLNTPH;"	 �������/HaqrhU<#�����YUW[ht����}th[YYYYYY��������������������UU[gt��������tgbab^Urnnu��������������tr�����  �������������#/HQV[a]PH<2/&#�����������������������

�������)5N^\`YN@>@5)#*/3<HNU\\UH<6/(#RVZ\anz�������zna]WRqlru��������������tq[WV[[\hhkh`[[[[[[[[[
#.2<JMTUQ3/(
!#/36<@BC></,'##!426:>BHgox}xtg[NB84���$56@)	������=979<BN[gtuqpjf_[NB=rtztrqpt}���������tr����)49>@41"���#)25BEJLNPNB5/$���������������������������������������������������������ptu��������|tpppppp|�����������||||||||0-.*%
��������
#0������������%)+5ENURNB5)^]]abgmvz������{zma^�����-6;A??6��IOQ[ahpty���{uth^[OI�����������������������
##
	�����������������#)/15<@C<8/# ��������������������������������������������������������������~�z�x�z���������F�S�_�l�x���x�t�l�k�_�S�F�E�:�/�:�C�F�F�I�Z�c�qŔŹŷŨ�z�n�b�A�0��������0�I�"�/�3�4�1�/�"���!�"�"�"�"�"�"�"�"�"�"����������������������Ʒ���������������̹����	�	��������ܹعϹ͹Ϲܹ߹��l�y���������y�n�l�e�l�l�l�l�l�l�l�l�l�l������������������������������������������������������������������������������������������������������������������ŭŹ����������������������ŹŭŨŤŤŨŭ�����������������������������������������������ʼҼּ�ּμʼ���������������������"�.�6�.�-�"���	���	��������6�9�?�C�\�`�\�S�O�J�C�6�*���
���,�6�O�\�h�k�j�h�\�T�O�N�O�O�O�O�O�O�O�O�O�O�лܻ޻����������ܻٻл̻Żлллм������ʼммʼ����������������������������������ĿѿҿѿĿ������������y�~�������B�N�O�[�i�t�x�~�t�p�g�`�V�O�L�D�A�B�T�a�i�����r�a�T�;�"���������	��/�E�T��5�>�C�A�E�T�U�A�(�����������������������������������������������������5�A�Z�s���~�s�i�_�N�A�(����	���(�5�B�O�[�Q�>������������������������)�B�����4�:�;�8�9�4�(���
���������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���������#�&�����������������������ĚģĦĳĳĵĳĳĦĢĚėĒĔĚĚĚĚĚĚ�����#��������������������	�"�G�g�r�t�r�m�`�T�;�/�����;ɾо�	��"�(�/�0�/�*�&�"���	��������� �	����������������������������������������������'�4�@�D�Y�b�d�Y�@�'����������f�s����������s�i�f�d�f�f�f�f�f�f�f�f�4�M�a���������������s�f�Z�4�,�(�$�)�4���	��"�(�'�"��	�����׾Ѿʾ׾���������=�V�`�V�>�0�$��������������������/�H�\�_�U�;�"�	����������������������/�ѿݿ�����%�,�/�3�(����ݿѿͿƿɿ��A�D�N�V�Z�g�s�x���������s�g�Z�N�J�A�7�A�T�a�c�o���������z�a�T�/�"�������	�"�;�T���������������������������������������˼ʼּ������	�����ּӼʼǼżƼʼż�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��O�[�h�h�h�[�W�O�B�;�B�M�O�O�O�O�O�O�O�O�����������������������������������������������������������������������������s�f�\�U�X�f�s��������������������������ɺغںɺ������������~�|�|�u�}���)�5�B�[�i�t�t�j�N�9�)��������)ŇŠŭ��������������ŹŭŠŔőŇņŃŃŇ�!�:�F�Z�_�a�Z�R�F�=�:�-�!�������!�-�1�:�A�F�M�S�T�S�M�F�:�-�!� ��!�"�+�-��4�9�4�1�$������ܻĻ����ûܻ�����{ŇŉŌŌŇ�{�n�b�Z�a�b�n�w�{�{�{�{�{�{¿����������������¿²ª¦¥¦ª±²¼¿ǔǡǧǦǡǙǔǈǂ�{�o�m�o�s�w�{ǈǊǔǔ �   H G h J 7 L 8 : L - H / _ o @ V \ G l / e N 4 �  8 $ c ' d K  F P D r @ ` N J B > Q 4 � Y G  5 < 8 6 C � 4 : j    f  �  �  �  Z  �  �  #  t  �  x  �  A  �  p  f  ?  �  B  v  �  r  �  �    ;    J    �  �    :  �  �  1  2  c  �  �  b  #    m  �  �  E  z  _  �  	  �  �  R    �  �  ?  �1�ě��D��=����o;D��;ě�;�`B;�`B<u<e`B=#�
<T��<�t�<D��<�o<e`B<���=o<ě�<�/=�o=���<ě�=�{=D��=\)<�h=}�<�<�=�+=\)=H�9=�hs=C�=�%=<j=�C�=u=�hs=<j=��=Y�=�t�=q��=<j=<j=T��=�E�=ě�=��m=� �=���=�{=�"�=��=��=�Q�B
�B�B&�DBfhB��B�`B 3ZB��B�B�:A���B<-BQ�B"3oB!�$B.B�BD�BX�B.�B��A��RB3>B*Bd B	�\Bb�B�SB�|B�xB��B��B�qBNB��BZkB�BD�B�(B�EB��B
v�B�kB<�B�BV?B��B
LpB7�B$lB�cB�aA�0�B�,B��BT�B��B�B{B
�B��B&��B@�Bv�B�B >�B��B�YB�EA��CB?�BA[B"?�B!��B�B��B@lBD
BZB��A�t+BחBH�B@�B	��BB�B�&B�BJ.B�MB=�B��B/XBA�B?0B>�B?�B��B��B@}B
�"B�9BD�B6B@�B�B
O�B?�B$?�B�NB��A���B?�B�SB��B�B�)B½A�|ZA���@���A��vA�TB�5?�OA	[A�+<A�p=B�A�9�Ar�S@�w�A^�B |B��@��@�gAs�A��A��|A��A���A��A�gA3t�C�ȁA��UA�9�A�)5A_&A�vDA��L@�AC��AA^�AX�B	�A�zUA��-A�uA�դA��OA�7C�p�A��A04BAs� AF@Z"A�ځA�h@v�@{F�@��A�({A��,B͏A���A��
@�m�A���A���B�S?N��A�A�]�AЅ�B��A��NAs�@�=A]��A�2JB��@���@�)Ar�A�{�A���A��A��uA��PA�g�A2+cC���AҀ�A�w�A�.Ad�qA���A���@�>IAD�AB׿AX��B��A�o�A���A��A��/A�vfA��C�j�A�<hA0��As�AD��@ZA�o4A�b@tDc@xa�@�M�A��A��aB�            P         
               !      
                  	      /   B      B         	   )         (   	      -            #      "   	   V      !               )   (   A      3                           ?                                                      %   5      #   %                  +         #      '      !   /         -                            !            !                     )                                                         1         %                  '               !         )                                                !         N��N��N���P�cN�@N���N���N��N\c�N��NmRVN��NW2N���NM�%O%8NԍN�x-OuO ��N��PO�V�Pe�bN�B�O��O�[�Ox�N76O�oN�ߖN�B/O���O�Oh�O�nAM�nkO���O�O���O�hOkl�N��dO���O)]�Ol��N�>�M��NL��N2��O���O#XQOv�jOCBO���N�kQOTL�N�I�N�u\N�  [  Y  �      l  1  %  �  8     �  �  �  �  (    �  �  �    w    Y  	�  �    �      �  D  �    �  i  W      %  �  r  
l  �  �  �    �  n  �    I  �  �  [  	   g  �  l��j�u��`B<ě��ě���o�D��:�o;o;�`B;�`B<�C�<o<#�
<#�
<D��<D��<T��<�o<�o<�t�=o<�j<�t�=\)<��
<�1<�1=+<�j<�j<��<�/<�h=+<�=�P=+=�w=��='�=�w=�t�=�w=0 �=,1=0 �=49X=8Q�=ix�=�+=�O�=�+=�hs=�t�=��w=���=���=�{~���������������~~~~��������������������::;<@IPUXVUSI==<::::����)5BU^ZI5���������������������������������������������������������������@?BBOTTOKB@@@@@@@@@@��������������������������������������fdcfmoz{}{zmffffffff����������������������������������������������������������������������������

�����!%)58<5-)!!!!!!!!!!05679@BOP[XROGB600001/,6BEJOU[_`[YOB>861!')+268?B<>)��������������������	"/;CEILKH@;"�����/H^hpmeU<#����YUW[ht����}th[YYYYYY��������������������UU[gt��������tgbab^Uoortz�������������to�����  �������������"##/<HKQUWUOHA<2/'%"�����������������������

�������
	)5OWYUNC=;<5)
#*/3<HNU\\UH<6/(#TX\anz��������zna`YTosty��������������to[WV[[\hhkh`[[[[[[[[[#*/<GIPQL</#
!#/36<@BC></,'##!>969>BN[gkuy{ysg[NB>�����!0,������?:99:>BFN[gmnhd\[NB?srt���������~ttssss������)143.'"�#)25BEJLNPNB5/$�����������������������������������������������������������ptu��������|tpppppp|�����������||||||||������
#%##
��������������������%)-5BGMNQMB5)aaacfmz�������zmbaaa�����-6;A??6��OR[bhrtw����ztth`[OO�����������������������
##
	������������������#)/15<@C<8/# ������������������������������������������������������������|�����������������_�l�x�~�x�q�l�g�_�S�O�F�:�F�S�T�_�_�_�_�0�<�I�U�\�f�zŉœő�{�b�U�<������0�"�/�3�4�1�/�"���!�"�"�"�"�"�"�"�"�"�"����������������������Ʒ���������������̹����	�	��������ܹعϹ͹Ϲܹ߹��l�y���������y�n�l�e�l�l�l�l�l�l�l�l�l�l������������������������������������������������������������������������������������������������������������������ŭŹ��������������ŹŭŬũūŭŭŭŭŭŭ�����������������������������������������������ʼμּݼּͼʼ���������������������"�.�6�.�-�"���	���	��������6�9�?�C�\�`�\�S�O�J�C�6�*���
���,�6�O�\�h�k�j�h�\�T�O�N�O�O�O�O�O�O�O�O�O�O�лܻ޻����������ܻٻл̻Żлллм����ʼμμʼ������������������������������������ĿѿҿѿĿ������������y�~�������N�[�d�g�t�w�|�t�o�g�a�W�P�N�N�H�N�;�H�T�^�h�k�r�m�a�T�H�;�/�"�����"�;���4�>�=�B�Q�Q�A�5������������������������������������������������������(�5�A�N�^�g�e�`�Z�N�A�5�(�������(�B�O�[�Q�>������������������������)�B����'�4�7�8�6�6�(������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������������������������������ĚģĦĳĳĵĳĳĦĢĚėĒĔĚĚĚĚĚĚ�����#��������������������	�"�G�c�o�q�m�`�G�;�3�"�����Ӿξվ�	��"�(�/�0�/�*�&�"���	��������� �	����������������������������������������������"�4�<�@�Y�T�M�@�'������������f�s����������s�i�f�d�f�f�f�f�f�f�f�f�Z�����������������s�f�Z�E�7�-�+�4�@�Z���	��"�(�'�"��	�����׾Ѿʾ׾�����������0�B�E�C�7�0�$�����������������/�;�H�Z�]�\�P�;�"�	����������������"�/�ѿݿ�������"�(�'����ݿѿϿɿ̿��g�s�v���������s�g�Z�N�L�G�N�Z�\�g�g�g�g�;�H�T�a�j�m�u�y�z�r�a�T�H�;�/�"���%�;���������������������������������������˼ּ������
��������ּʼɼǼǼȼͼ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��O�[�h�h�h�[�W�O�B�;�B�M�O�O�O�O�O�O�O�O��������������������������������������������������������������������������������������������s�f�c�Z�^�f�s������������ɺͺʺɺ������������~�}�~�����)�5�B�Z�g�k�g�c�M�B�5�0�)�������)ŠŭŮŹ����������ŹŭŠŔŔŊňŔŞŠŠ�!�:�F�Z�_�a�Z�R�F�=�:�-�!�������!�-�:�A�F�L�S�T�S�L�F�;�:�-�!�!��!�$�-�-��4�9�4�1�$������ܻĻ����ûܻ�����{ŇŉŌŌŇ�{�n�b�Z�a�b�n�w�{�{�{�{�{�{¿¿����������������¿²«¦¦«²²¿¿ǔǡǧǦǡǙǔǈǂ�{�o�m�o�s�w�{ǈǊǔǔ �  T R h J 7 L 8 < L  H , _ o @ V X G N $ e N  �  8  c ' k K  F P B r % T A 9  > N 4 � Y G  ) / 4 6 @ � 4 4 j    f  �  �  �  Z  �  �  #  t  <  x    A  �  p  f  ?  �  ?  v  9  v  _  �  1  ;  �  J  9  �  �  �  :  �  �  1  �  c  3  %  �  �  3  m    �  E  z  _  �  i  �  =  R  �  �  �  "  �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  [  V  Q  L  G  B  =  8  3  .  -  2  6  ;  ?  D  H  M  Q  V  <  G  P  V  Y  X  T  P  H  ?  4  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  P  �  �  �    �  �  �  q  &  �    �  (  �  }  �              
  
  
  
    %  7  I  \  n  �  �  �  �  l  d  ]  T  G  ;  (    �  �  �  �  i  @    �  �  w  A  
  1  #       �  �  �  �  �  �  �  �  �  �  �  &  [  �  �  �  %        �  �  �  �  �  �  �  �  �                �  �  �  �  �  �  �  �  �  �  �  s  d  T  D  4     �   �   �  %  ,  3  6  6  +       �  �  �  v  J    �  �  _     �   �     �  �  �  �  �  �  }  _  >    �  �  �  �  X  .     �   �    c  �  �  �  �  �  �  �  �  |  G    �  �  D    �  �  �  �  �  �  �  {  e  P  ;  $  
  �  �  �  �  �  d  D  $     �  �  �  �  �  �  �  �  �  y  f  R  @  3  &          �  �  �  �  �  �  ~  x  r  k  e  _  [  Z  Y  X  W  V  U  T  S  R  (  '  &  &  %  $  #  !                 3  ^  �  �  �              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  C    �  q  +  �  �  [     �  �  �  �  �  �  �  �  r  `  O  7    �  �  |  D    �  �   �  �  �  �  �  �  �  �  �  i  Q  ;  %    �  �  �  }  Q  #   �                 �  �  �  �  r  A  
  �  �  �  �      �  �  "  J  e  t  v  j  J    �  {    �  @  �  E  �  �  Y  �        �  �  �  i  1  �  �  �  O  �  \  �  �  L  u    Y  U  Q  L  C  9  .  #    
  �  �  �  �  �  �  y  =     �  C  �  	?  	j  	�  	�  	x  	]  	0  �  �  N  �  ~    �  �    ~  l  �  �  c  ,  �      ,  E  G  M  T  T  F  /  �  o  �  ^  �  	      �  �  �  �  �  �  �  �  q  [  ?    �  �  b     �  �  y  n  c  W  K  <  -      �  �  �  J  �  �  P    �  �  �  T  �  �  �      �  �  �  �  �  z  3  �  B  �  �  /  f          �  �  �  �  �  �  a  C  .                �  �  �  z  f  R  9    �  �  �  �  b  5    �  �  N     �  !  A  C  7  $    �  �  x  (  �  �  /  �    �  o  =    �  �  u  a  L  7    �  �  �  �  ]  :    �  �  �  v  /   �   R  �          �  �  �  �  �  k  ;    �  �  �  �  �  ?  �  V  v  �  �  �  o  T  %  �  �  y  @  !      �  �  �  �  |  i  d  `  [  V  Q  L  G  B  =  6  -  $      �  �  �  Z  *  5  M  U  V  Q  G  9  &    �  �  �  �  �  \  -  �  r    �        �  �  �  �  �  k  N  ?    �  �  <  �  �  =   �   �  �  	            �  �  �  �  d  1  �  �  y  "  �  z  S  �    #  %    	  �  �  �  �  �  �  s  3  �  �  �  Z  �  |  U  �  �  �  �  y  a  @    �  �  �  K  �  �  <  �    $    _  g  o  n  g  a  Z  S  M  G  B  >  :  =  @  H  S  \  d  k  �  �  	�  	�  	�  
,  
S  
h  
f  
R  
3  
   	�  	6  �  �  �  �  
  X  �  y  X  4    �  �  �  r  J  "  �  �  �  �  �  c  %  �  �  L  c  �  �  m  P  2    �  �  �  h  #  �  �     �  �  .  �  �  �  �  �  �  k  K  (    �  �  �  s  K  #  �  �  �  P  �    !  :  S  m  �  �  �  �  �  �  �  �  �  �      /  A  R  �  �  �  �  �  �  }  z  w  u  n  c  Y  N  D  9  .  $      n  `  S  D  4  $      �  �  �  �  �  �  �  o  ]    �  p  �  �  �  �  �  �  �  p  P  (  �  �  W  �  <  �  /  �  A  �  �  �            �  �  �  �  �  d    �    o  �  �    
d    ;  I  ;    
�  
�  
�  
f  
,  	�  	�  	*  �  %  �  �  �  �  u  j  o    s  `  I  0    �  �  �  ]    �  )  �  4  �  <  �  �  �  �  �  �  w  _  7  �  �  U  �  �  ?  �  �  ?  �  �  A  T  X  Q  E  5       �  �  �  w  K  !  �  �  �  �  C    	   �  �  �  �  �  x  L  &  �  �  I  �  �  %  �  v  =  ,  �  g  O  5       �  �  �  y  S  ,  �  �  �  c  +  �  �  w  9  C  �  s  [  B  #     �  �  {  I    �  �  a  *  �  �    �  l  ^  O  A  1      �  �  �  �  �  ~  X  2    �  �  I  