CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�V�t�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M낢   max       P�$      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �#�
   max       >hs      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Q��R   max       @E�33333     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vP�\)     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�=�          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >aG�      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�t�   max       B,m�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,�h      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >oQ�   max       C�g�      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Bm�   max       C�hd      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          !      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M낢   max       O�v      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�qu�!�      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >hs      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Tz�G�   max       @E�33333     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @vP�\)     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @=   max         @=      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_خ   max       ?��PH�     `  U�   .      �               8               '      !   2   \                   b      &   
   (         #      2                                       	                                    G   
   
         O���N�;Pn�N�FN[MfM낢N�P%�UNN�;N35N8�O3U�OI��N�^O��Oք�P�$N��NU@�Op�O���O�O�y2N���O�^�N|��O��N�6�O�O��lN���O�߼N�|N`}�N�uO�N�WORZN�@�Ol-�O	�iN`"lOpC�OdDN�EOQLN:��O��6N4��O7:/N �N[��NtN�&<O��N�K�O#�<N���N��	N�5tNe}�N�L�#�
��C���o��o�T���D���ě���o;D��;D��;�o;��
<t�<#�
<D��<T��<e`B<e`B<e`B<u<���<��
<���<���<���<���<�/<�`B<�h<�<�=+=+=C�=t�=t�=t�=t�=�P=�P=��=#�
=<j=D��=H�9=L��=P�`=aG�=aG�=e`B=e`B=q��=}�=�o=�t�=��P=��=�E�=��>1'>1'>hsHGEDHQ[gt������tg[NH��������������������$5Nt�������qgNB5$rtwttt�����trrrrrrrr����������������������������������������x}����������xxxxxxxx%!)BO[h~��z{thOB6.)%��������������������oqt�����ytoooooooooo#(jemruz{�����������mj���������������������
��������������#0LUWT@0#
�������
/>B<<5/#
���NHBEOt������������gNLLN[gtz����~tmhg[NLL���������������������������������������� ���"$*-25BQTPPB5)% ���
#/>JLLH>/#
 � 
#%'&#
����������������������������������������UXahksz����������naU��������������������155;BJNX[bgjkgg[NB51//3<HUanz����znaT<8/�����������������������).-)($�������������������������������
����������)5;=95)����������������������������������������������������������������zyxz��������������z#/;<=HRRNH</#�����������������������������

������("%()469BOZ[a[YOB6)(������������������������������������������������������������������������������� ���������� �����������ZX[_hosph[ZZZZZZZZZZ��������������������./<HOHHHIH<02/......~}����������������~~\_d`amz��������zmaT\���������������������������������������������������������ymhghmz���zyyyyyyyyy���������������������������������������������������������������������
�����
��������������������N�P�[�[�[�N�J�B�>�;�B�J�N�N�N�N�N�N�N�N�)�6�H�N�N�J�B��������ìãéñ������)�zÂÇÒÓÕÕÓÇ�z�r�t�z�z�z�z�z�z�z�z�����������������������������������������0�=�C�I�N�I�=�0�0�0�0�0�0�0�0�0�0�0�0�0���������������������������������������޾��ʾ۾����־������Z�M�9�5�V�\�t�����������������������������������������������Y�e�o�p�e�Y�L�J�L�N�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�;�H�T�V�Y�T�H�;�;�:�;�;�;�;�;�;�;�;�;�;�
��(�0�<�G�I�O�I�H�<�6�0�#���
��	�
�h�tāčęĚĦĩĪĦğĚčā�t�h�a�[�[�h������������ùøìçìïùÿ�����������ż���������������������Y�D�;�@�D�Y�f�z����5�B�P�[�a�]�@�5�)��������������������Z�s���������s�Z�J�A�5�$��
�
� ���H�T�V�[�_�X�T�K�H�;�6�/�&�/�;�<�>�C�H�H�����������������������������������������/�<�@�H�U�a�k�zÅÆ�z�n�a�Y�@�<�/�$�#�/�"�.�G�T�m�w�}�y�`�T�G�;�.�"���
���"�;�G�T�`�m�y�������y�m�`�T�G�I�I�G�;�8�;�׾���"�.�2�:�5�!�	���׾Ⱦ��������ʾ׿`�m�y���|�y�q�m�`�T�L�G�@�G�T�V�`�`�`�`�A�M�f�s�}�z�u�v�~�s�e�Z�M�;�4�/�-�.�4�A��������������������������������������������(�5�N�[�a�^�W�N�5�(���
��������������������������s�j�g�s������U�b�b�n�u�{�}�{�z�n�h�b�U�J�I�A�@�D�I�U�������������������������������àìïöùúùìà×ÓÏÏÓÖÙàààà���ûлٻ߻�ֻлû������s�l�b�j���������!�-�:�F�P�Q�F�:�-�!�����!�!�!�!�!�!�zÇÌÇÄ�z�u�n�a�^�a�i�n�r�z�z�z�z�z�zƎƚƧƳƸƿƳƫƧƠƚƎƊƍƎƎƎƎƎƎ������������������������������������������� �	�
���
�����������������������������(�4�6�>�A�4�(�������������ϹϹù¹����������ùϹܹ���ܹйϹϹϽ��������ĽƽҽٽؽнĽ����������|�z���EEEE E'E*E*E*E(EEEE D�D�E EEEE������������������������������������������"�;�H�M�H�@�9�#���	���������������������������ºɺպʺ������������������������������������������������������������������������������������������������������Ŀѿܿۿݿ�ݿٿѿĿ������������������	��"�'�;�D�H�E�;�/�"���������	�	������	��������������	�	�	�	�	�	�U�\�T�S�H�;�/�"���	��	���"�/�;�H�U��'�1�4�7�4�'����������������� �������������������²´½½²¦¦ ¦­²²²²²²��!�-�.�6�:�@�:�-�+�!��
�� �����Š����������������������ŹŭŠœōŉŔŠ���������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D}D{DuDsDtD{D{�����������������������������������������zÀÇÓàèèàÓÇ�z�z�z�z�z�z�z�z�z�z²¿������¿²¦¦«²²²²²²�t�z�t�k�g�[�[�[�g�o�t�t�t�t�t�t�����!������������������ 1 P ' S 0 O u P 5  3 7 5 b .  3 [ Y K : a 6 E ; ] J 6   I M ? G } E [ � K s 1 R : C P E > \ V K J  / F  G : ' | : A S    C  N  �  $  `    `    j  E  J  �  �  �  E  �  �  !  �  �  5  �  &  �  3  �  s  �  U  j  �  �  �  �  �  k  �  �  �  �  V  p    c    H  �  O  q  �  2  p  n    ;    e    �  �  g  �;��
�u>aG��49X�o�t���o=]/<o<�o;�`B<�9X=D��<ě�=8Q�=�%=�
=<��
<���=0 �=P�`<�=��#=t�=�%=t�=�+=��=D��=�o=C�=���=#�
=D��='�=49X=#�
=m�h=49X=�%=�o=D��=�7L=q��=m�h=�%=]/=�t�=}�=��=�O�=��=���=��T=ȴ9=�E�>��=ȴ9=�G�>t�>z�>&�yB	D�BX�B��B
�B~B�B
��B��BN�B�fB�B -�B�:Bx@B%�BCB
��B	IeB!�;B�KB�=B1B�B6^B�	B��B�HB�BNBO0B!o�B/(B(�B*lBNB1�BbB!X�B =�By<BOdB!�5Bx�B)�B,m�BZkB��B�BG�A�t�B�wB K�B�5B�|A��B ��Bu�B��A�1�BˊB@BQB	@dBCB�0B
@_B1�B?�B
�gBכB?�B�aB�>B <�B��B�AB%��B;�B
�*B	��B"+rB'�B��B�B�-B:�B�B��B��B�BL�BBCB!�B>�B�(BƂB?�B��B@GB!F�B��B8�B��B!� BF�BEsB,�hBE�B�B!MB@A��B��B E_B,lB�SA���BBB�|B4�A�^}B��B@BC�A��A�.RA�i�A�SIA䞊B
°A��AJ�A��?�P	A��eA�aAݚ1A�[�@�{A�3�A��A�J)@���A�;�Ac��Ah��AV��Ail�A>҅A��A��*AF�A�)A��A˚�@���@w7�A���B�dB��A� zA3��>oQ�A"�C�g�@��dA��@X�A 'A���Az A��MA��A���@�F@��A��.@im�A��A�vGC�ŧA�5*A�NxA��A���A2��A�s9A���AӓA�]}A�VB
�8A��AJ��A�q�?�xA��A�SA�~6A͉�@��A���A��JA�r)@��[A�ȟAb�XAi%WAV�gAj�@A>�hAЄ�A�
MAE;�A�rA�y�A�Y7@���@v�UAȆ�B�%B�QA��A3�>Bm�A#��C�hd@��8A�w�@%lkA
+A���Ay�3A���A�|�A�J�@���@���A��@d�A���A��~C�ȕA�v�A�j�A�FKA�DA2��   .      �               8               (      "   3   \            !      b      '      (         #      3            	                  	         
                                    G   
   
                  /               -                     '   !   7                  #                           !                                                                                                                                       !                                                                                                                                             OZ�N�;O���N�FN[MfM낢N�O�ҴNN�;N35N8�O&5N�I!N�^O��*O}PO�vN��NU@�O9ՓOc(O�OK�"N���O�^�NE��OhEN�6�O�NJ�3N���O^a�N�|M��SN�uO�N�WOE��N�@�OE>�O	�iN`"lOF�gOdDN�EN���N:��O5� N��O.�N �N[��NtN�&<O��N�K�O K�N�9�N��	NItNL��N�L  N  �  �  �  �  �  �  �         �  :  �  �  �  �  �  �    V  �  �  
  �  T  �  q  F  T  !  x  Y  ^  a  �  �  �  K  S  t  .  "  �  �  i  #  4  P  �  �  �  	  -  �  �  P    u  �  �  ������C�=����o�T���D���ě�<�9X;D��;D��;�o;�`B<���<#�
<�1<�h=}�<e`B<e`B<�t�<ě�<��
=��<���<���<�/=C�<�`B<�=T��<�=49X=+=#�
=t�=t�=t�=�P=�P=#�
=��=#�
=D��=D��=H�9=Y�=P�`=q��=e`B=ix�=e`B=q��=}�=�o=�t�=��P=�j=�Q�=��>	7L>	7L>hsNNR[gt}����{tig[YQNN��������������������348BN[gty{yvng[NB;63rtwttt�����trrrrrrrr����������������������������������������x}����������xxxxxxxx649=BO[hmqssnh[OB<86��������������������oqt�����ytoooooooooo#(immuz{����������zrmi�����������������������
�������������#0=HJE?70#
��������
 )/23/+#
��gecekt�����������togLLN[gtz����~tmhg[NLL��������������������������������������	����"$*-25BQTPPB5)% 
#'/8=>;0/#
 
#%'&#
����������������������������������������lfinnvz����������znl��������������������456<BLNT[bgjjge[NB84:9<EHLU_[UH<::::::::��������������������������&$#�����������������������������
����������)5;=95)����������������������������������������������������������������}z{|���������������}#/;<=HRRNH</#�����������������������������

��������("%()469BOZ[a[YOB6)(��������������������������������������������������������������������������������	��������� ������ZX[_hosph[ZZZZZZZZZZ��������������������./<HOHHHIH<02/......~}����������������~~\_d`amz��������zmaT\������������������������������ ����������������������������ymhghmz���zyyyyyyyyy�����������������������������������������������������������������
����
��������������������������N�P�[�[�[�N�J�B�>�;�B�J�N�N�N�N�N�N�N�N����)�.�/�,�����������������������zÂÇÒÓÕÕÓÇ�z�r�t�z�z�z�z�z�z�z�z�����������������������������������������0�=�C�I�N�I�=�0�0�0�0�0�0�0�0�0�0�0�0�0���������������������������������������޾������ʾϾվϾ�����������t�r�w�������������������������������������������������Y�e�o�p�e�Y�L�J�L�N�Y�Y�Y�Y�Y�Y�Y�Y�Y�Y�;�H�T�V�Y�T�H�;�;�:�;�;�;�;�;�;�;�;�;�;��#�#�0�;�<�I�C�<�0�/�#���
��	�
���tāĂčĚĜĚĚčā�t�m�h�o�t�t�t�t�t�t������������ùøìçìïùÿ�����������ż�����������������r�f�Y�Q�I�N�Y�f�r������5�B�N�P�T�M�B�5�)���	����������5�A�N�Z�g�q�|�~�{�s�g�Z�N�A�5�(� �#�(�5�H�T�V�[�_�X�T�K�H�;�6�/�&�/�;�<�>�C�H�H�����������������������������������������a�g�z�|ÄÄ�z�t�n�a�]�U�C�=�<�2�<�H�N�a�.�;�G�O�`�k�s�m�g�`�T�G�.�"�����"�.�;�G�T�`�m�y�������y�m�`�T�G�I�I�G�;�8�;������	����������׾ʾʾȾ̾׾�`�m�y���|�y�q�m�`�T�L�G�@�G�T�V�`�`�`�`�A�M�f�s�}�z�u�v�~�s�e�Z�M�;�4�/�-�.�4�A��������������������������������������������(�5�N�V�[�Y�P�N�A�5�(���	�����������������������s�j�g�s������U�Y�b�n�t�{�|�{�y�n�g�b�U�K�I�B�A�E�I�U�������������������������������������àìïöùúùìà×ÓÏÏÓÖÙàààà�����������ûл׻ӻлû��������x�v�v�����!�-�:�F�P�Q�F�:�-�!�����!�!�!�!�!�!�zÇÈÇ�~�z�n�n�a�a�a�k�n�u�z�z�z�z�z�zƎƚƧƳƸƿƳƫƧƠƚƎƊƍƎƎƎƎƎƎ������������������������������������������� �	�
���
�����������������������������'�4�<�@�4�(��������������ϹϹù¹����������ùϹܹ���ܹйϹϹϽ��������ĽϽнսнĽ�������������|����EEEE E'E*E*E*E(EEEE D�D�E EEEE�����������������������������������������"�/�;�B�<�5�/�"� ����������������	��"�������������ºɺպʺ������������������������������������������������������������������������������������������������������Ŀѿܿۿݿ�ݿٿѿĿ������������������"�#�;�<�A�D�@�;�1�/�%�"����
��
��"�	�
����
�	���������	�	�	�	�	�	�	�	�	��"�/�;�H�S�[�T�R�H�;�/�"���	��	����'�1�4�7�4�'����������������� �������������������²´½½²¦¦ ¦­²²²²²²��!�-�.�6�:�@�:�-�+�!��
�� �����Š����������������������ŹŭŠœōŉŔŠ���������������������������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDwDzD{D{�����������������������������������������zÀÇÓàèèàÓÇ�z�z�z�z�z�z�z�z�z�z²¿¿����¿²¦¦¦­²²²²²²²²�t�x�t�m�g�\�g�q�t�t�t�t�t�t�t�t�����!������������������  P  S 0 O u , 5  3 8 1 b 0 (  [ Y / 3 a & E ; j : 6  F M 6 G � E [ � I s 3 R : : P E 3 \ X C I  / F  G : # n : $ L    R  N  4  $  `    `  �  j  E  J  :  �  �  ^  �  ]  !  �  �  �  �  �  �  3  �  �  �  >  n  �  �  �  d  �  k  �  �  �  �  V  p  �  c    �  �  �  :  �  2  p  n    ;      �  �  Z  F  �  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  @=  G  �  �  �  .  G  N  F  3    �  �  �  Y  �  p  �  �  �  �  �  �  t  i  ]  Q  F  :  .  #           �   �   �   �   �   �  �  �  �  �  �  �  L  �  �  �  �  �  F  �  �  �  �    	l  �  �  �  �  �  �  �  �  �  �  �  �  �  }  m  \  K  /    �  �  �  �  �  �  �  }  n  ^  N  =  +       �  �  �  �  ]  8    �    j  U  @  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  r  n  l  i  g  d  a  _  \  Z  W  8  �  �  �    `  �  �  �  �  �  �  g  A    �  H  �    �        �  �  �  �  �  �  �  �  �  �  �  {  t  w  |  �  �     (  /  5  9  <  =  A  D  E  <  ,    �  �  �  �  �  r  i    �  �  �  �  �  �  �  �  �  �  �  �  �  z  e  P  ;  '    �  �  �  �  �  �  y  `  B    �  �  �  n  B  '    $  :  O  j  �  �  
    *  2  2  9  #  �  �  Y    �     #    �   �  �  �  �  ~  h  \  O  A  5  %          �  '  �    �  �  `  �  �  �  �  �  �  �  �  �  �  �    D  �  �  �  I  �  F    S  �  �  �  �  �  �  �  �  �  c     �  Y  �  >  e  ]   �  �    <  t  �    @  h  }  �  �  |  T    �  )  t  �  ]   �  �  �  �  �  �  �  �  �    o  e  `  [  V  N  F  9    �  �  �  �  �  �  �  �  �  �  v  g  X  H  K  Z  i  x  }  �  �  �  �  �          �  �  �  S    �  �  7  �  �  ]  *    �  -  A  P  V  R  F  1    �  �  �  T    �  �  C  �  �  i  ;  �  d  D  3  "      �  �  �  �  �  �  �  �  h  M  3    �    �  �    I  p  �  �  �  �  J  �  �    
�  	�  �  N  �  �  
    �  �  �  �  �  �  �  �  n  \  J  #  �  �  �  �  �  �  �  �  w  e  K  (  z  p  _  @    �  �  }  E  �  |  �    �  S  T  W  n  �  �  y  b  K  4      �  �  �  �  j  G  $    �  �  �  �  �  �  �  �  �    W  $  �  �  N  �  r  �  �  �  q  k  d  _  Y  T  M  E  <  1  %      �  �  �  �  K    �  E  F  D  <  0      �  �  �  u  7  �  �  b    �  �  ]    r  �  �  �  �    �  �  �    :  K  Q  <  �  �  �  �  �     !  
  �  �  �  �  x  G    �  �  �  �  �  p  j  h  f  e  c  T  h  m  s  x  v  f  O  /    �  �  l  N  )  �  �  @      Y  T  O  I  D  >  8  2  +  %          �  �  l  (  �  �    +  4  <  D  L  R  X  ^  S  S  _  e  j  !    q  d  W  L  a  Z  T  M  H  E  C  A  ?  ?  ?  ?  :  1  (        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  L  *  �  |    �  �  �  �  �  �  �  �  w  j  _  V  L  B  9  )      �  �  �  �  �  �  �  �  p  V  8    �  �  �  e    �  �  (  �  S  K  2    �  �  �  �  X  "     �  �  �  �  �  �  j  2  �  �  L  P  R  Q  L  F  ?  4  "    �  �  �  �  �  l  5  �  W  �  t  9    �  �  h  8    �  �  m  9  �  �  _     �  �  !  d  .         �  �  �  �  �  �  �  �  t  g  [  O  C  *  �  �    !  "                �  �  �  �  �  o  Y  l  n  Z  B  �  �  �  �  �  o  [  F  .    �  �  �  �  �  U  $  �  �  2  �  }  z  m  _  N  =  &    �  �  �  �  �  �  o  V  >  #      E  ^  e  h  g  d  ]  S  H  =  8  4  0  )      �  �  �  #          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ,  .  0  3  4  .      �  �  �  z  R  (  �  �  F  �  �  V  @  F  L  N  G  A  2    
  �  �  �  �  �  g  G  (    �  �  �  �  �  |  T  0    �  �  �  L    �  1  �  -  �    �  �  �  �  �     �  �  �  �  �  �  �  �  �  r  c  S  B  1      �  �  o  [  L  =  -  "        �  �  �  �  �  �  {  f  P  	  �  {  B  �  �  z  :  �  �  u  4  �  �  �  �  �    2  +  -    �  �  �  �  b  >  "    �  �  x  C    �  �  �  g  B  �  �  |  h  X  p  f  T  B  3  %    �  �  �  {  '  �  X  �  �  �  �  �  n  Q  2    �  �  �  ^  +     �  �  a    �  �    9  M  M  ;    �  �  X  �  '  q  �  �  
�  	�  |  .  �  Q        �  �  �  �  _  D     �  �  �  �  �  �  �  �  �  �  u  J       �  �  �  s  T  9  "    �  �  �  y  `  F  /    �  �  �  �  �  �  �  �  g  O  :       �  �  �  �  s  �  �  �  �  �  �  �  �  �  �  �  m  <  	  �  �  h  0  �  �  �  �  �  �  �  �  m  O  -    �  �  i  ,  �  �  T  �  �  .  �  V