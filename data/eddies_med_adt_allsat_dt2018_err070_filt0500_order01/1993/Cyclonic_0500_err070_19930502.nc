CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�O   max       Pc(       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��C�   max       <��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F�p��
>     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?θQ�     max       @v{33334     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P`           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�            7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       <D��       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�c�   max       B4��       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�`�   max       B4��       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?UY   max       C��t       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?.ȹ   max       C���       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          W       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�O   max       PF �       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��3���   max       ?�d��7��       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       <��
       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @F���
=q     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?θQ�     max       @v{33334     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P`           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?�c�	�     �  \8   	      -                  !         L            +         %         9      *         
         *               	   A   Q            	         W   /   	      /          (   	   
         	                              	         O(�N�qiO���O-��N��Nw<M�OO���O��O6��N�P�O�S�NT9�NfOO��P^�O�R\NJ�FO�"O�,OF�nPA��OV��O���O�|Nj�*O��Nɠ�N|�PV�rO��O���NU�"OW�N�d�P�1Pc(OQ��O%nO��]N��O�� OCh�P7��P'��N�/�O��~Oº�OChN[%�P�N�H�OPSO¨�ND`#O�O"��N�2JN�&O?�NZD�NN�O�xM�D�N�c�N�k�N���N�pN��N<��
<T��<o%   ��o�o�o�o�ě��ě��ě��ě���`B��`B�#�
�49X�D���D���u��o��o��C���C���t����㼬1��1��1��1��1��9X��9X��9X��9X��j��j��j�ě��ě���`B��`B��h�o�+�+�C��C��C��\)��P��P���,1�,1�,1�,1�8Q�@��@��D���H�9�L�ͽT���aG��u�}󶽁%��%��C�����������������������������������������)6BOYcgc[B,' ��������������������669COUROJCB614666666#''#mnuuxz}znnljmmmmmmmmrx����������������ur�����������������������)5=?:5)����&/7<HIOIHB<1/$&&&&&&��������������������#,/00/$#"inz~����znijiiiiiiii%0<Zbn{���{jhbUI<4,%����#0IbgbUI&�������������������������#)/9<A<</#!���"#���������3;HLT\afca^WTHEC=833#/<?HIJGFA</$#!Vaz�������������rfXV���
#*+#
�����������	��������������������������������������������������������������������������������������������FHIUabfgda`UOHFFFFFF5?bu��������nYI<0*+5MNSU[gt�~���tge[NKMM������%"��������56>BGOU[OB7655555555����������������������������������������r�����������������tr������2+������������ �������������������������������()/BN[gpsg_[XNB5)##(��������������������6<BJfkmhjggdhg[B9856DHUXanpzxonfaUHGA=?D���������������������������������ggqt���������tgggggg���������������������������������������������������������������������)BN[ehlaTB5) 	
"%&" 	Y[bit���������og`[WYeipt������������tgee��������������������\anz���������zxlca\Y\cgot{�������tgf[ZY��������������������lnuz}������zzynnllllrz���������������zur(/2<GHKH?<8/,)((((((����������������������������������������otw�����troooooooooo����������������~}}���������������#/0/./0/-+#-/<HOOPH=</.///&----MOP[ehjptuvth][ROMMM�������������������
���#�$�#���
�����t�n�g�a�g�t¦¨§¦�_�W�S�W�W�\�g�x�������������������x�l�_�����������(�0�4�9�@�A�A�4�(�������׾۾������	��	���������ìçéìùþ������ùìììììììììì�A�=�A�N�Z�_�g�l�j�g�Z�N�A�A�A�A�A�A�A�A��׾��������������ʾ׾����������������������������$�0�4�;�?�?�7�0�$�����#�*�0�=�I�V�Z�]�Y�V�M�I�=�0�&�$����������������������
�����������������弤������������ʼּ��������ּʼ�����������������������������������������������������������������������������������������ݽнǽƽнݽ�
�(�4�A�M�R�P�A�(��н������u�r�������������ݽ�����ݽݽ������������������*�6�C�O�h�o�h�X�6���H�=�<�:�<�=�H�U�U�Y�W�U�H�H�H�H�H�H�H�H�`�T�G�;�=�G�`�y���������������������y�`��������(�5�<�A�N�S�X�N�A�5�(���6�/�/�4�6�@�B�O�[�h�t�~�y�t�l�h�d�[�O�6������������������"�,�4�A�E�E�5�������������������������������	���	��������'�$�/�5�A�Z�����������������s�Z�N�A�5�'�������������������$�*�,�*�#����[�X�O�J�G�O�[�h�i�j�h�]�[�[�[�[�[�[�[�[���ݿؿտݿ�����������������I�B�C�I�U�[�b�n�s�{ŀŀ�{�x�n�m�b�U�I�I�z�p�m�b�l�m�z�������������~�z�z�z�z�z�z�A�5����ȿ�����(�N�t�|���������s�Z�A�m�k�`�T�L�I�J�T�`�e�m�o�p�x�y�����y�m�mÓÐËÇ�n�g�h�l�n�zÓÛìöúûòìàÓ�a�_�U�S�U�X�a�n�x�u�n�h�a�a�a�a�a�a�a�a����������(�4�A�G�M�Z�c�]�M�A�(���T�R�?�;�.�+�%�.�8�;�G�T�`�a�m�q�r�m�`�Tč�y�[�O�D�F�F�R�hāĚĦİĬĥĥġĜĜč��ƻ����û˻�����@�M�Q�\�`�a�^�@����m�`�Q�L�F�G�T�`�m�y�����������������y�m����������������������ʾ̾ӾоʾǾ����������������������������<�J�G�<�*�����������������	����	�	��������������.����������.�;�T�`�y��u�`�G�;�.�������������������������������޺ɺ������������ֺ��!�:�H�Y�U�F�)�����ɼf�f��������ּ���!�1�5�/����ּ����f�	���������� ��	���� ����	�	�	�	�	�����۾ԾԾ׾����	��.�>�:�.�"��	���������(�5�A�Z�s�|���{�o�g�N�A�(�������'�3�@�L�Y�d�a�Y�W�H�@�3�'���Z�V�X�Z�g�s�������v�s�g�Z�Z�Z�Z�Z�Z�Z�Z�U�J�H�J�I�L�UÇÓìù������ùÓÄ�z�n�U�����������������������������������������#���
�����������#�)�3�0�<�@�<�8�0�#��¿¦¡¡¦²����������
��������¦¨¦FF
FE�E�E�E�FFFF!F$F.F1F2F1F$FFF��������������������)�+�-�*�)�����
�����������������$�0�0�$��������y������������������������������������������������Ŀѿݿ�޿ݿؿӿѿʿĿ�E�E�E�FFFFFF$F&F$FFFE�E�E�E�E�E��x�t�l�k�j�l�x�|�������������x�x�x�x�x�x�T�R�S�T�\�a�m�z���������������z�m�k�a�T����������������������������������������������������������	������	�������׽S�R�I�G�B�G�S�`�l�w�w�y�~�y�l�`�S�S�S�SE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EiEfEaEdEiEuE�E�E�E�E�E�E�E�E�EuEiEiEiEi�ܹӹϹȹϹӹܹ�������� ������ܹ� > V M + p I � <  q A 2 . / ] [ v 8 \ O % A H = " 1 = c K l H a : I  ? : \ B \ h s % U h Y N K H 6  B B 6 4 .  u b > L _ X C j ; b 6 W    +    �  p  z  9  R    1    �  [  n  x  *  D  �  i  �  j  �  �  �  V  "  �  F    �  I  1  �  t  �  =  �  �  �  o  �  �  �  �  �  �    c  �  �  |  x  �  �  �  ]  ?  a  1  �  �  �  q  ]      �  �  �  �<D��;o��P��j���
��`B���
�e`B��㼛�㼛�㽧D���#�
�#�
�]/�ě���t��P�`��9X�C�����o�q����h��`B����/��/��%��h�,1��P��P�o��-��������w�D���t��,1�q����F�����0 ŽY����T��+�#�
����<j�P�`��hs�8Q�L�ͽ�\)�y�#�aG���O߽e`B�aG�����ixս�C���hs���罙����{BުB��BӫB�B0�7BXB��B4��B�zBaB�_B��B��BB&�$B$٫B!B0<B-̬A���B1[BFB��B��B8�B ��B[:Bu�BܾB(�B	3mB��Bc6B�-B��B��B�B�wB!��B��BµBH�B�/B�B-,#B
N�B��B�B��B��BesA�c�B

�B
��B�4BN�B	�uB&�B\�Bp B��B!2?B��B�BO8BhyB�%B�2B��B-�BC�B3�B�wB0��B��B��B4��B��B�;B��BħB��B�4B&��B$�B �;B��B-��A�y;B��B{�B�B��B@B1B=7BEOBʠB)I7B	{BPB@�B��B�lB��BCSB�NB!��B~FB��B��B�"B��B->NB	�mB@yB�lB>BJWB��A�`�B	�?B
@	B8
B7`B	�OBǴB@�B�IB�HB ½B��B��BAB?�BÌB��B�rA�ĽA��@��{A5/�AW��A��A� PAO�B	bSB
ڟA�/@��A��)A��A6�A"��A���Aĥ�Aj�1A��A��IA���A�6�A�/A��A�Z�A�Q�A�v7A��A���Aj(
A�DMA���A7�AgȐAܘ�@�PAm.9AMf(A��.A�[�Aa~>A�x@R`�Ad�A\�AZ9�A�3?��tA��vA�+�A��SA�A��>A� �C��tA�9lB��AI�Ax'�C��@@�%A�=A���A�lA 
C�IoC��	?UYA�@�A��@��NA5 
AU.A�_�A���AOUB	a/B>�A�b�@��A��A�|:A7�:A"��A���Ać�Ak�A��A�J�A�sA��:A�u]A�~�A�{�A�u�A�kFA�@wA��&Aiz�A�oAƀ�A7xAh$,A�|n@��Ao�eAMUA���A��AcpA��N@K#�A^�A[�A[�A�| ?���A���A��dA���A�yKA��A��C���AԔ�B�AH��Ax��C���@��A���A�%�A��A�GC�S�C��?.ȹ   	      .                  !         M            ,         %         9      *                  +               
   A   R            	         W   /   
      0          (   	   
         	         	                  	   
                                                      '   3   #      %         -      %                  9                  %   3         !      )      3   7         #         %         !                                                                                                               %                        '                     /               )      +   +                                                                        O(�N���OY�O��N��Nw<M�ON�'�O�"N��N�P�OmNT9�NfON��>O~��O���NJ�FO=��O�,OF�nO��iO,"�O�XO�|Nj�*N̚�Nɠ�NL;O�^�O��O���NU�"N�k�N<�Oq:�PF �OQ��N���OgO�N��O�� N�D�P��P	��N�/�OUɏOmm�OChN[%�O�D�N�H�OPSO�%�ND`#O�O�}N_]�N�&O?�NZD�NN�N�=�M�D�N�0DN�k�N���Ns"�N��N  *     [  �  �  |  �  �  �  x  J  :  �  �  3  �  o  =  r  !    �  (    �  �  �  �  �    m    �  	  8  �  	�  �  �  *  �  �  R  	  H  �  �  >    �  Y     r  �  �  *  �  f  "  m  U  �  3  
  �  =  '  '  �<��
<D���D���o��o�o�o��`B����o�ě��\)��`B��`B���ͼ�/�T���D����h��o��o��󶼛���㼛�㼬1��j��1��9X�\)��9X��9X��9X��/�����L�ͽo�ě����+��`B��h���49X��w�C����<j�\)��P�H�9���,1�<j�,1�,1�@��H�9�@��D���H�9�L�ͽY��aG��y�#�}󶽃o�����C�����������������������������������������6BMTWXWQOB<61)��������������������669COUROJCB614666666#''#mnuuxz}znnljmmmmmmmm����������������������������������������)05;=85)&/7<HIOIHB<1/$&&&&&&��������������������#,/00/$#"inz~����znijiiiiiiii9<BIUZ]]\URI?<;99999�
#09<??90$
�������������������������#)/9<A<</#!�����������3;HLT\afca^WTHEC=833#/<?HIJGFA</$#!joz��������������{qj�����
#()#
���������� �������������������������������������������������������������������������������������������GHKUaaefaUQHGGGGGGGGCIRbt~����������nUICMNSU[gt�~���tge[NKMM������%"��������56>BGOU[OB7655555555�����������������������������������������������������������������))�������������� �������������������������������&)-/5BN[c[WSNKCB5*&&��������������������6<BJfkmhjggdhg[B9856HHUadmmia_UHFB@BHHHH���������������������������������ggqt���������tgggggg���������������������������������������������������������������������)5BNVY^^[VNB5)	
"%&" 	Y[bit���������og`[WYgjkt������������tgfg��������������������\anz���������zxlca\egstt�������tig[[^ee��������������������lnuz}������zzynnllllrz���������������zur(/2<GHKH?<8/,)((((((����������������������������������������otw�����troooooooooo�������������~}��������������#/0/-/-*#(/2<HKNNH<2/((((((((MOP[ehjptuvth][ROMMM�������������������
���#�$�#���
�����t�p�g�g�g�t¦§¦¦�t�t�t�t�_�\�\�a�l�x�����������������������x�l�_�����������(�.�4�7�>�;�4�(��������׾۾������	��	���������ìçéìùþ������ùìììììììììì�A�=�A�N�Z�_�g�l�j�g�Z�N�A�A�A�A�A�A�A�A�������������ʾԾ׾ھپ׾ʾ����������������������������$�/�0�6�6�0�+�$���0�+�+�-�0�4�=�I�V�X�[�W�V�J�I�=�0�0�0�0���������������������
�����������������弱�������������üʼּټ�����ּʼ����������������������������������������������������������������������������������������
���(�4�>�A�E�A�<�4�(���������������������������Ľнֽ޽ӽĽ����������������������*�6�C�O�\�d�T�6���H�=�<�:�<�=�H�U�U�Y�W�U�H�H�H�H�H�H�H�H�T�N�I�I�S�T�`�m�y�������������y�x�m�`�T��������(�5�<�A�N�S�X�N�A�5�(���6�/�/�4�6�@�B�O�[�h�t�~�y�t�l�h�d�[�O�6�������������������	��"�0�3�3�.�&��������������������������������������������N�M�G�F�I�N�Z�g�s��������~�v�s�g�Z�N�N�������������������$�*�,�*�#����[�X�O�J�G�O�[�h�i�j�h�]�[�[�[�[�[�[�[�[�����ݿۿؿݿ��������������������I�B�C�I�U�[�b�n�s�{ŀŀ�{�x�n�m�b�U�I�I�z�v�m�d�m�m�z���������|�z�z�z�z�z�z�z�z�N�A�5�%������������(�N�Z�f�l�f�^�N�m�k�`�T�L�I�J�T�`�e�m�o�p�x�y�����y�m�mÓÐËÇ�n�g�h�l�n�zÓÛìöúûòìàÓ�a�_�U�S�U�X�a�n�x�u�n�h�a�a�a�a�a�a�a�a��������&�(�4�A�M�S�P�M�A�4�(�#��`�Z�T�G�E�G�G�T�W�`�m�l�`�`�`�`�`�`�`�`�e�[�V�Q�Z�c�h�tāčēĖĔĒĐčĈā�t�e��ѻ����ڻ���4�@�M�Y�\�[�R�M�-�����m�`�Q�L�F�G�T�`�m�y�����������������y�m���������������������ʾ˾ʾǾ�������������������������������/�6�C�D�B�6�*����������������	����	�	��������������.����������.�;�T�`�y��u�`�G�;�.��������������������������������޺�ֺɺ��������ֺ����:�J�I�F�2�#���⼋�����������ּ����*�0�(����ּ������	���������� ��	���� ����	�	�	�	�	�������۾߾����	��"�.�2�,�"��	������(�5�N�Z�l�n�j�g�Z�N�A�5�(��������'�3�@�L�Y�d�a�Y�W�H�@�3�'���Z�V�X�Z�g�s�������v�s�g�Z�Z�Z�Z�Z�Z�Z�Z�a�[�V�V�V�V�a�n�zÇÓàì÷÷ìÓ�z�n�a�����������������������������������������#���
�����������#�)�3�0�<�@�<�8�0�#����¿©¦¦¢¢¦²¿�����������������¦¨¦FF
FE�E�E�E�FFFF!F$F.F1F2F1F$FFF������������������)�*�,�)�)���������������������������������������������y������������������������������������������������Ŀѿݿ�޿ݿؿӿѿʿĿ�E�E�E�FFFFFF$F&F$FFFE�E�E�E�E�E��x�t�l�k�j�l�x�|�������������x�x�x�x�x�x�T�S�T�V�]�a�m�z���������������z�m�i�a�T�������������������������������������������������������	�����	�	������������S�R�I�G�B�G�S�`�l�w�w�y�~�y�l�`�S�S�S�SE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EuEjEiEeEgEiEuE�E�E�E�EEuEuEuEuEuEuEuEu�ܹӹϹȹϹӹܹ�������� ������ܹ� > Y F ) p I � &  I A " . / & G l 8 . O % 8 ;  " 1 2 c A Y H a : I R 4 7 \ : Y h s  M X Y K M H 6 ! B B < 4 .  B b > L _ X C f ; \ 6 W    +  �  �  G  z  9  R  �  !  /  �  D  n  x  �    f  i  �  j  �  B  }  @  "  �  �    _  Y  1  �  t    _  �  k  �  �  
  �  �    �  �    �  �  �  |  e  �  �  c  ]  ?  6  n  �  �  �  q  A    �  �  �  �  �  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  *      �  �  �  �  �  �  �  ~  i  X  Z  ]  c  j  X    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  +  ;  H  X  Z  Q  <    �  �  x  3  �  �  �  ?  �  A  �  �  �  �  �  �  �  �  �  �  o  V  2  
  �  �  S    �  �  �  �  �    u  j  a  \  V  Q  K  A  2  $        	        |  y  v  s  q  o  m  k  j  l  m  n  �  �    O  R  G  <  0  �  �  �  �  �  �  �  �  �  �  y  h  W  D  /       �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  9     �  �    7  Z  u  �  �  �  �  �  �  f  =    �  d    �  9  �    S  r  w  p  d  T  A  ,    �  �  �  �  b  0  �  �  Z    J  E  >  6  )    �  �  �  �  i  8    �  �  `  '  �  �  �  	�  
J  
�  
�  
�    .  9  2    
�  
�  
>  	�  	   ~  �  )    �  �  �  �  �  y  b  J  2      �  �  �  �  �  j  S  <  &    �  �  �  {  m  _  Q  H  C  >  8  3  -  ,  2  9  ?  F  M  S  �  �  �  �  �  ~  �  �  &  1  2  $    �  �  G  �  j  �  g  �  �  �  Z  z  �  �  �  �  �  �  �  {  O     �  S  �  �   �  V  f  l  c  \  U  N  8    �  �  �  ~  V  1  
  �  �  �  Q  =  ?  A  B  @  =  ;  5  -  %        �  �  {  E    �  �      *  ?  T  a  j  p  p  m  b  M  2    �  �  C  �  �    !      �  �  �  �  �  �  �  �  �  x  q  j  b  Z  M  @  2        �  �  �  �  ]  ,    �  �  �  }  Z  D  G  @  �    i  �  �  �  �  �  �  �  �  �  \    �  {  )  �  c  �  0  Q        $        �  �  �  �  �  j  P  ;  @  9  5  X  �  x  �  �  �                  �  �  �  �  <  �  �  Z  �  �  �  �  �  �  �  }  k  Y  D  -    �  �  �  �  m  .   �  �  �  �  �  �  �  r  _  K  8  $      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  M    �  �  d  �  �  �  �  �  �  �  �  �  �  x  ]  C  .  $            �  �  �  �  �  �  �  �  �  {  m  `  S  ?    �  �  �  V      �  �  �  �  �      �  �  �  �  S    �  a    �  �    m  e  ]  V  S  O  L  J  G  F  D  B  >  :  1  #    �  �  �      	  �  �  �  �  f  :    �  �  �  {  S    �  b  �  [  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  !  �  �  7  �  �  �  �  �  �  �  �         �  �  �  �  �  �  z  b  L  ,  �  �  �  �  �  �    .  3  -    �  �  �  �  z  T  .    �  �  }  !  �  �  -  _  �  �  �  �  v  <  �  i  �    R  ~  >  �  	m  	�  	�  	�  	s  	U  	)  �  �  G  �  h  �  k  �  �  ,  w  w  �  �  �  �  �  �  p  _  N  3    �  �      �  �  �  �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  R     �  �  9   �    "  #       &       �  �  �  �  y  A    �  �  �  �  f  �  �  �  �  u  _  d  h  _  U  A  #    �  �  o  5  �  �  �  �  �  �  �  �  �  _  8  '    �  �  �  �  �  �  p  .   �   �  �    &  C  Q  Q  P  J  A  4       �  �  c  �  ~  �  I  �  �  �  	  	  	  �  �  �  �  �  �  Q  
  �  b  �  1  3  �  &  �  ;  =  <  #    �  �  �  N    �  w  %  �  d  �  y  �   �  �  �  �    p  \  G  -    �  �  �  �  c  @      �  �  �  �  �  �  �  �  �  �  �  �  �  t  P  "  �  �  O  �  �  y  ]  *  7  =  <  =  =  7  *      �  �  �  O    �  �  Y  �  �        �  �  �  �  �  �  �  W    �  a  �  u  �  5  o  �  �  �  �  �  �  �  �  �  {  g  T  A  .       �   �   �   �   �  ~  �  �  !  A  U  Y  T  F  ,    �  �  �  O    �  5  �       �  �  �  �  �  �  i  Q  8      �  �  �  �  i  3  �  �  r  m  i  ]  Q  B  1      �  �  �  �  b  4    �  �  
  .  �  �  �  �  �  �  �  ~  n  �  �  |  Y  )  �  �    �  /  �  �  �  �  �  �  �  x  n  f  ]  T  K  C  <  9  6  2  /  ,  )  *         �  �  �  �  �  �  k  O  2    �  �  �  �  x  L  �  �  �  �  �  �  �  �  �  �  z  h  L  &  �  �  v  '  �  �  <  �    e  a  Z  N  <  !  �  �  �  \  3            !  "    �  �  �  �  �  �  �  �  �  �  �  �  h  Q  ;  &    �  m  `  P  :  #    �  �  �  �  c  1  �  �  ]    �  I  �  �  U  @  *    �  �  �  �  �  h  H  '    �  �  �  y  e  X  J  �  �  �  �  �  �  �  e  L  7  "    '  `  �  �    a  �  �  1  2  0  )    �  �  �  �    �  �  �  �  �  ^  3    �  �  
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  t  j  �  �  �  �  �  �  �  y  Q  '  �  �  �  _  &  �  �  �  C   �  =  <  ;  9  6  5  6  *    �  �  U  <    �  �  W     �   �    &  !  	  �  �  �  �  m  A    �  y    �  k    �  T  0        $         �  �  �  �  �  �  k  P  3    �  �  r  �  �  �  t  Y  :    �  �  �  �  �  y  `  D  &  �  Y  �  1