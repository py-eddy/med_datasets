CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�^5?|�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�b�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       ;��
        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @Fu\(�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @v�\(�     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O@           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�%@            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       �o        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B2`_        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�6�   max       B2eo        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�l�   max       C��i        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C��        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          h        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��l        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����>C   max       ?�Ƨ-        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       ;��
        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @Fp��
=q     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
>    max       @v�\(�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @N�           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�!�            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dv   max         Dv        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?Ӵ�3��        W�         U         g   >   	            
   .         1   	                           ;            0   
         ,      J   !                                                                       
            N��nNM�P�b�N�;nN��P�7�P��YO��O��pN�kM��NJ~�P0IN�LO	{�O�+�Nf��O��,N�'O���N{��Op(�N��gN��OC�O�80N���N{��O��`PCO]NxZ�OfN���O=��N��EP`�OT�%Nl'�N�,�OL�nN���OV�O�@�O�FNO�O���O��,N2��N�sN�S*N3��N���N���O"|N��N��N���N�s�O�߾N�c�N96SO�N�TN��>;��
;�o;o��o�o�o�o���
��`B�o�D���e`B��C���C���t���t����㼛�㼛�㼴9X��j���ͼ�����/��/���o�\)����w�#�
�0 Ž49X�8Q�@��D���P�`�P�`�Y��]/�e`B�ixսixսixսu�y�#�y�#��%��%��o��7L��\)��\)��hs������P���P��9X�ě����ͽ��ͽ�xս�xվ� #(./3<AB></#      OOX[hiih^[OOOOOOOOOO��#0b������^UH<#���#)/221-)#	yz~����������zyyyyyy��<bjkaa[I<0	����v����	��������uvLNU[gtzzwlg[WONKGHKL���)5BKNMN)���������������~������_admpmma`^__________6BNOVTTOKDB666666666HTam���������maTGABHNOS\bhuw����uqh\OONN#<?HU]UPH@</#���������������{zvw�FHUW_`]UMHGHFFFFFFFF����&#��������')46BBKO[`[OJB6340)'egt������������utg`e����������������������������������������#).566665)'"######��������������������`an|������������zne`%/<HU`ghejUH</#��������������������%)46ABFHBBB65-)'%%%%x��������������{vssxL[g�������������gNJL`abnz{}zvnea\]``````������������������������

�����������������������������egkt{��������tjgeeeez��������������yuuz#)/<=??HJSTHF</*#! #Xanz~�~zpnaUXXXXXXXX+6BO[]a`[[POMB?64*++��������������������X[dhntxvth[UQSXXXXXX������

��������������

���������vz��������������{xxvgn{��������{sngbcgz�����������������zz*5BNt��������t[B4(����������������������������������������������������������������� ����������������������������������FHLSUamnxtnaZUIHFFFFst�����������|zwvtssmt��������~tljmmmmmm��

������������

��������zz����}znaWURU\ajnzz��&'# ���������������������!& ����������������������������������������ntv�����������tnnnn���	������� �	��"�#�+�)�"�����������������������������������T�H�3���4�s�����������������˺ɺɺźɺκֺ�������������ֺɺ������������������������������o�n�}�������4�H�B�3�!�(���彷���y�`�D�5�2�<�G�`�y���Ŀѿ׿ҿԿ���ܿ��y������������(�4�>�A�N�R�N�A�5�(������s�g�a�[�X�]�g�s����������������������������������������� ���������������[�Y�[�b�h�t�u�w�t�h�[�[�[�[�[�[�[�[�[�[�a�]�a�b�n�zÇËÇ�z�y�n�a�a�a�a�a�a�a�a�����������������������	�"�D�?�,������˾����������������������������������������6�*�)�'�(�)�0�2�6�=�B�G�O�R�P�P�O�N�B�6�M�D�=�@�M�Y�f�r���������������r�f�Y�M����������������������������������������ÓÇ�z�u�w�u�zÇÓÛàåò÷þ��ùìàÓ�m�i�`�\�`�`�m�p�y�����������������y�q�m��������������'�-�0�/�(�&�����U�R�H�<�;�6�<�C�H�U�[�Z�[�U�U�U�U�U�U�U��ؾʾ�������s�i�s����������������	��	���"�/�;�A�A�;�/�"��	�	�	�	�	�	�.�$�"���"�.�;�G�J�Q�G�B�;�.�.�.�.�.�.�H�F�?�<�6�7�<�D�H�L�U�_�a�h�b�f�a�[�U�H�M�A�3�/�0�6�A�M�Z�s�������������s�f�Z�M�I�H�I�U�X�b�n�{�ŇōŇ�{�n�b�U�I�I�I�I�����������������ʾʾʾ���������������������ĺĳĬĳ�����
��#�'���
�����������x�g�[�Q�N�O�t¦²����������¿���������������������������������������Y�N�L�I�I�L�R�Y�b�e�r�~�������~�r�e�a�Y����������������������������������������a�V�U�L�M�V�a�g�n�zÅÇÉÌÌÇÂ�z�n�a�V�Q�I�?�=�9�1�=�I�V�_�b�d�d�b�`�V�V�V�V�ѹֹ���3�<�L�Y���ɺκƺ������L�3�����FE�E�E�E�E�E�F-F1F=FJFRFVFZFQFJF=F1FFF=F;F8F9F=FJFVF[FVFUFNFJF=F=F=F=F=F=F=F=�����������������������������������������O�C�6�*�'�)�'�*�6�C�O�\�]�h�o�t�h�\�R�O���������'�(�(�*�'���������ݿѿĿ������ĿͿѿۿݿ���� ��������5�(���(�5�A�N�Z�g�s�����������s�Z�N�5�������������	��"�/�;�/�*�"��	�������׻������������ûлܻ���������ܻлû�����ƳƚƎ�}�r�l�oƁƧƳ���������������˾����������	��"�.�4�8�8�1���	������������������������������(�5�A�N�Z�g�i�g�Z�Y�N�?�5�(��t�o�h�d�`�^�h�tāāĆăā�w�t�t�t�t�t�tùóôùú����������ùùùùùùùùùù�$�������������$�0�3�:�0�(�$�$�$�$E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Ƴưƭƽ����������������������������Ƴ��������������������������������D�D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�E*E*EEEE&E*E7ECEJEPEYE\E_E\EPECE7E*E*���������Ŀȿѿҿֿ׿ѿ̿Ŀ��������������������������Ľݽ�����
����ݽн������S�Q�G�B�:�:�1�:�G�H�S�`�k�l�p�l�`�\�S�S�������!�(�.�2�.�!���������������������'�*�0�'�%��������������������ûλлܻ�ܻлû�������������ýùóììììù���������������������� ; P R A m 3 - p D / P W = ] a 3 Z l I 6 F x g 3 c  a r F J X ) b * B } g b E $ > 5 Q C G Z Y a d R e K B b [ X R A c T u " z F  �  :  �  !  V  y  d  �  �  �    �  3    {  �  �  �    �  �  �  �  �  �  �  �  �  �  Q  x  C     �  �  �  7  �  $  �  �  �  m  G  T  �  E  [  =  �  �  �  �  �  �  �  �  �  �  �  :  #  i  ��o�D������ͻ�o��
=��o�T���ě����
�e`B��j�}󶼼j�\)��7L��`B�8Q���ͽ�w��P��P�+�C��<j��9X��P�#�
����� ŽH�9��%�aG���E��aG����#��1�aG���+������7L���P���㽲-��7L��Q콾vɽ�7L��hs���-������罼j��-���w��Q������`��h��G���G��
=q������BNPBF�B&|FB�B L�B%�/B+�B��B��BN�A���BSjA��LB2`_BkaB�~B��B�)B8�B
�B!	,B � B�+BӎBߛB��B��B��B-�B
�TB�dB!9�B#�B��B
�B_:B�B�Bp:B`�B��B�gB��Ba�B(�RB��B	B4B �B��B�B��Bf�B�B
��B
XB�	B�LB��B�!BB��B�;B�|B
��BB@B@B&�CB��B ��B%@B*B��Bl>BCuA�6�B3A���B2eoB;�B��B��B8�B�B
�2B �YB!@,B��B�B�6B�4BΓB�B �ZB9zB�B!@�B�B�cB	�5BQUB=XB��B��BN�B�:B�&BQB?�B(��B��B	N�B �xB�BtsB�lB?�BF�B
��B	�~BA�B��B�pBK5B�rB�[BB��B
��A\�h@�CDA��M@D��AӢ�A)IgAn��A��A���A���Aۇ�A��:A��fALhA���@��WA��IA���AmB�A2�A��DAN�A��~Aa�lA�P�A?!�A�~ALu�A��A��A���?�ȫA�
}Aǡ�Bu�?�r�C���C��iA��B �?�l�A}k�A�d�A���@��kB\�A\2�A���A�BFA�7�A��B	u�C�/B�`A�
eC�+�C��4Ax�A(�A�As�@�2s@��	A�[�A\��@��@A��!@Dj�A�z�A$��An�YA���A�P A�|�AۏQA�yA���AK��A�y@��0A��<A�}�Am�A4�AĀ+AN�A�|�AbՙA�|A=yA�q�AK�|A�l�A��)A���?�DA�|BAǇ�B�m@'�C��C��A�|�B?��A{S!A�XA��!@���BBAZ��A��A��A�$%A�vuB	TC��B7#A�J�C�'�C���Ax��A'9CA�A
��@��-@��(A�c         U         h   ?   	            
   .         2   
                           ;            0   
         -   	   K   "                              !      	                        !                              G         ;   9      #            -         #                  #                        -                  ;                     !         %   !                                    !                        =         3   %                  %                                                   '                  9                                                                                    N��nNM�P��lN�6FN��PH��O��O��O�QrN���M��NJ~�O�&N���N��\Ok�Nf��OjZfN�'O<1N^QO�vN��gN��N��OY WN���N{��O��`PPNxZ�OfN�7
OWNN��EPPJO�Nl'�N��OL�nN_�NOV�O�+nO��O�OC�O}MN2��N�sN�S*N3��NDN�N���O"|N��N��N�;.N�s�Ok�N�c�N96SO�N�TN��>  ;  $  �  �  c    o  �  �  }  �    �  F  #  g    �  Z  �  �  �  �  1  s  �    �  Y  F  7  �     
d  �  �  	Z  6  �  7    j  .  �  K  �  ]  �  �  p  }  �  �  Z  �  �    �  J  �  �  �  m  x;��
;�o�o��`B�o���
��/�ě��t��t��D���e`B��/��t������h���㼴9X��������ě���/������/���<j�o�\)���<j�#�
�0 Ž8Q�P�`�@��P�`�e`B�P�`�]/�]/�m�h�ixսq���q���u��C���\)��%��%��o��7L�����\)��hs������P���㽴9X�Ƨ���ͽ��ͽ�xս�xվ� #(./3<AB></#      OOX[hiih^[OOOOOOOOOO�	#<n������bSI<����),/.+)yz~����������zyyyyyy�
#<Ubd\VVOI<0# ����������������������MNW[^gtxxvkgd[UNIILM)5BHLJLB)����������������������_admpmma`^__________6BNOVTTOKDB666666666HMTamz�������zmaTIFHU\fhru����uih\WPUUUU#/<>HUZUNH</#|����������������{|FHUW_`]UMHGHFFFFFFFF�������������')46BBKO[`[OJB6340)'qt�������������zuliq����������������������������������������#).566665)'"######��������������������enz�������������znge"%+/:<HUY__\UH</$#""��������������������%)46ABFHBBB65-)'%%%%x��������������{vssxa���������������[RPa`abnz{}zvnea\]``````������������������������

�����������������������������egkt{��������tjgeeeez���������������{wwz#,/8<>ACFGF@</,%#!!#Xanz~�~zpnaUXXXXXXXX,66BO[[__[ZOBA66,,,,��������������������U[hltwtth[XSUUUUUUUU������

���������������
��������wz��������������}yywgn{��������{sngbcg��������������������<P[gt�������tk[NEB:<����������������������������������������������������������������� ����������������������������������FHLSUamnxtnaZUIHFFFFst�����������|zwvtssmt��������~tljmmmmmm��

�������������

���������zz����}znaWURU\ajnzz��%&%"���������������������!& ����������������������������������������ntv�����������tnnnn���	������� �	��"�#�+�)�"���������������������������������Z�I�7�7�J�s�������������������������ֺҺɺɺɺҺֺ�������������ֺֺֺ������������������������������y�x�������н���4�;�5�$��	���н������m�`�N�F�D�I�T�`�y�����������¿��������������������(�1�5�<�A�K�A�5�(��s�g�d�]�\�`�g�s�����������������������s����������������� ����������������������[�Y�[�b�h�t�u�w�t�h�[�[�[�[�[�[�[�[�[�[�a�]�a�b�n�zÇËÇ�z�y�n�a�a�a�a�a�a�a�a���������������������	��"�-�6�3�"�	���侥���������������������������������������6�,�)�'�(�)�)�2�3�6�:�B�F�Q�P�P�O�L�B�6�Y�T�J�G�G�H�M�Y�f�r�������������r�f�Y����������������������������������������àÕÓÇ�z�w�x�z�}ÇÓÜâïõûÿöìà�m�i�`�\�`�`�m�p�y�����������������y�q�m�����������������"�(�)�-�,�(������H�<�<�7�<�H�N�U�W�Y�Z�U�H�H�H�H�H�H�H�H�ʾľ������������������ʾ׾ܾ����׾��	��	���"�/�;�A�A�;�/�"��	�	�	�	�	�	�.�$�"���"�.�;�G�J�Q�G�B�;�.�.�.�.�.�.�H�C�?�<�;�;�G�H�Q�U�Y�a�d�a�_�b�_�Y�U�H�Z�M�A�;�4�4�4�=�A�M�Z�f�s�����|�s�o�f�Z�I�H�I�U�X�b�n�{�ŇōŇ�{�n�b�U�I�I�I�I�����������������ʾʾʾ���������������������ĺĳĬĳ�����
��#�'���
�����������~�r�h�b�g�t¦°¿����������¿¦���������������������������������������Y�N�L�I�I�L�R�Y�b�e�r�~�������~�r�e�a�Y�����������������������������������������a�]�U�P�R�U�[�a�m�n�z�~ÇÊÊÇ��z�n�a�V�Q�I�?�=�9�1�=�I�V�_�b�d�d�b�`�V�V�V�V��޺�3�=�Y�����ɺͺź������Y�L�3�����F
E�E�E�FFFF$F1F=FJFVFXFNFJF=F4F$FF
F=F;F8F9F=FJFVF[FVFUFNFJF=F=F=F=F=F=F=F=�����������������������������������������O�C�6�*�'�)�'�*�6�C�O�\�]�h�o�t�h�\�R�O�������'�'�'�)�'����������ݿѿĿ������ĿͿѿۿݿ���� ��������N�5�(�� �(�5�A�R�Z�g�s�����������s�Z�N�����������������	���"�,�$��	�������׻������������ûлܻ���������ܻлû�ƧƚƎƃ�z�r�u�ƁƎƚƧƳ��������ƶƳƧ������������	��"�)�.�1�.�(����	������������������������������(�5�A�N�Z�g�i�g�Z�Y�N�?�5�(��t�o�h�d�`�^�h�tāāĆăā�w�t�t�t�t�t�tùóôùú����������ùùùùùùùùùù��	����$�&�0�4�0�$���������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Ƴưƭƽ����������������������������Ƴ��������������������������������D�D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�E7E-E*EE(E*E7ECEIEPEXEVEPEGECEBE7E7E7E7���������Ŀȿѿҿֿ׿ѿ̿Ŀ��������������������������Ľнݽ��������ݽнĽ����S�Q�G�B�:�:�1�:�G�H�S�`�k�l�p�l�`�\�S�S�������!�(�.�2�.�!���������������������'�*�0�'�%��������������������ûλлܻ�ܻлû�������������ýùóììììù���������������������� ; P Q 4 m B + k B / P W % Z ` ! Z e I / D e g 3 j  a r F H X ) ^ & B z ^ b C $ < 5 N 8 G ? J a d R e H B b [ X P A W T u " z F  �  :    �  V  �    M  O  �    �  �  �  ]  �  �  3    �  �  �  �  �  T  �  �  �  �  �  x  C  �  J  �  �  z  �  �  �  z  �  0    T  �    [  =  �  �  Z  �  �  �  �  �  �    �  :  #  i  �  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  Dv  ;  1  &      �  �  �  �  �  �  �  �  �  �  �  �  ~  n  _  $                 �  �  �  �  �  �  j  4  �  �  �  S  �  �  �  �  �  �  �  g  &  �  �  �  :  �  l     �    T   �  x  �  �  �  �  �  �  �  �  �  _  0  �  �  �  B    �  :  �  c  V  I  <  /  "      �  �  �  �  �  �  �  �  �  �  �  �  @  �  �      �  �  �  �  �  �  �  �  �  �  }  !  �  
  �  �    &  :  L  X  d  n  n  c  P  4    �  �  �  >  �  *   �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  E  ,        +  k  �  �  �  �  �  q  \  E  (    �  �  �  �  X  &  �  �  a  |  }  ~  �  {  t  i  [  L  9  %    �  �  �  �  L    �  �  �  �  �  �  �  �  }  u  n  f  X  B  ,      �  �  �  �  �        �  �  �      /  5  6  4  .  *  '    �  �  �  �  �  �  �  �  �  �  �  x  B    �  �  �  �  O  	  �    {  �  B  D  E  F  >  4  *      �  �  �  �  �  �  �  |  m  ^  P     #  "        �  �  �  f  9  
  �  �  ]    �  Q    �  �  �  ;  [  f  _  G  +  �  �  l  3  �  �  �  e    |  �  �      %        �  �  �  �  �  �  u  \  B  '    �  �  �  R  O  y  �  p  P  $  �  �  �  �  c  8    �  �  �  "  �  q  Z  T  N  G  =  1  &        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  g  R  <    �  �  {  :  �  �  >    �  �  �  �  �  �  �  �  �  �  w  8  �  �  a    �  �  @  �  g  o  x  �  �  {  x  p  c  H  ,    �  �  �  �  b  F  <  @  �  �  �  �  �  �  �  �  �  w  \  B  ,    �  �  �  �  v  R  1  )  "      	  �  �  �  �  �  �  �  �  y  ]  B  +      3  O  e  s  o  ^  F  -    �  �  �  q  ?      *    �  �  �  �  �  �  �  �  �  �  �  �  e     �  n    g  �  �  >  ?    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  g  U  C  0      �  �  �  m  ?  Y  -     �  �  �  u  T  '  �  �  }  ?    �  {  .  �  �  �    8  C  E  7      �  �  ~  D  "     �  �  ;  �     �  �  7  0  )  $             �  �  �  �  �  �      �    1  �  �  �  z  c  L  1    �  �  �  �  U  &  �  �  �  \  .               �  �  �  �  g  >    �  �  h  %  �  �  c  &  
Q  
[  
a  
d  
`  
T  
E  
  	�  	�  	b  	  �  [  �  �    �  �    �  �  �  �  �  �  �  y  Y  7    �  �  �  �  ^  >  0  A  R  �  �  �  �  _  5  +  �  �  j  /  �  �  '  �    q  �  h  E  m  �  	  	Z  	T  	>  	  �  �     �  V  �    N  �  �  6  �  �  6  .  '           �  �  �  �  �  �  �  �  �  �  �    p  �  �  �  x  i  W  B  +    �  �  �  �  b  ;    �  �  B   �  7  "    �  �  �  �  �  �  �  v  S  $  �  �  W    �  M  �  �  �        �  �  �  �  V  *  �  �  �  c  "  �  �  0  �  j  [  Q  I  @  6  )      �  �  �  d  6    �  �  �  �  �    #  .  ,  #      �  �  �  �  �  W  &  �  �  �  t  R  �  �  �  �  �  �  �  \  :    �  �  �  _    �  H  �  f  �  g  K  D  >  6  ,  !      �  �  �  �  �  �  �  `  @  #     �  �  R  |  �  �  �  �  �  w  U  -  �  �  �  [    �  {  .  �  V  W  V  V  W  \  [  R  >    �  �  z  8    �  p  �  �  �  �  �  �  �  �  �  �  s  Z  A  %    �  �  �  �  w  \  B  '  �  �  �  �  �  �  �  �  |  f  M  /    �  �  �  [  +     �  p  e  W  F  +    �  �  �  �  `  5    �  �  �  �  _  ;    }  r  g  \  O  B  5  &      �  �  �  �  �  m  C  �  �  *  �  �  �  �  �  �  �  �  �  �  �  �  �  x  \  ?  !  �  �  �  �  �  �  �  �  �  [    �  �  `    �  �  C  �  =  �  �  U  Z  >  %  
  �  �  �  �  �  p  w  �  ]  '  �  �  %  �  �  J  �  �  �  �  �  �  �  |  z  x  w  v  t  r  p  n  n  n  m  m  �  �  �  `  ;    �  �  �  W  $  �  �  �  V  I  M  [  q  �    �    �  �  >  
�  
W  	�  	C  �    a  �  �  �    /  G  Z  �  �  i  H  *    �  �  �  r  G    �  �  k  '  �  y    �    I  >  6  .        �  �  �  �  �  �  �  s  @  �  L  �  �  �  o  t  u  d  S  C  2      �  �  �  �  g  1  �  7  �  �  �  �  �  q  R  3    �  �  �  �  .  �  J    �  �  \     �  �  �  ~  a  ;    �  �  �  S    �  �  1  �  q    �  E  m  m  n  n  o  l  _  S  F  :    �  �  Y    �  �  s  ;    x  U  .    �  �  �  v  N    �  y    �  k    �  !  �  