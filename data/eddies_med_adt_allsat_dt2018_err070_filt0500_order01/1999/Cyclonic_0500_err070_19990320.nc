CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�?|�hs     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�K�   max       P���     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <�j     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @F��Q�     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v���
=p     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�4          D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ȴ9   max       <49X     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4�|     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�o�   max       B4�     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ps�   max       C��     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >F_a   max       C��w     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�K�   max       Py�`     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?�?���     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�j     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?z�G�{   max       @F�z�G�     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @v���
=p     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @P�           �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�̀         D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F7   max         F7     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?{�u%F   max       ?�>BZ�c      P  g      
      
         8            
            
   	         5   S               
               $      5   /                  
               $                   	   ,      	                        .         
         /   -   &            	            	            NۖN�O9�4N�?oN�6N�2�O҃�OW��N3��Ow� O7�O7�iO���O[f�N��0Nd%�O!��O۴/P���P��Nk@�N�+6O 3oN���O"`~N�^�Ob��O��N{�hO�SO^O��O���O��P�-Nbh�P��N�f�N��Oc��N���O&{O�YP��N'�WOI O��N{�Ou�Nl�FO��!N첺N�b�NV?M�K�N��oN�n�N>\O��N+dGO�Q)NB�!N�xN�}�OueNЌ-O���O�hO��N�kO� N	��O3�N\��OONKG�N�0�O`�2NWXOv�N$��<�j<T��<#�
<t�<t�<o;�`B:�o:�o�o�D����o���
�o�#�
�49X�49X�T���T���e`B�u���㼬1��9X��j��j��j��j����������/��`B��h��h��h���o�o�o�o�+�C��C��C��\)�t��t���P��P��P���#�
�#�
�#�
�#�
�#�
�#�
�,1�49X�49X�49X�49X�8Q�<j�L�ͽP�`�P�`�T���Y��Y��Y��Y��Y��e`B�}󶽁%��o�����+��7L������������������������������� 
��������)+6>ACEEB60)% ~����������~~~~~~~~
	���#)��������~����������������vv~#.<HR_a]UH</#!��������������������!#,/HUX]\USQSH</,#!!46BOO[]dhc_[OB<:6/-4�� #'*+,++/.#� *6COZcguyu\YP6*amz������~~}xxvmaXXa��������������������9<GIUWaYUKIF=<999999��������������������w�������������vuuwxw
#N{������{b0

����������������������������������lnrtz��������znnnllY[^ht��������th_[WYYyz������������zpyyyyRT[amqywrmlija]TMLORimz���������zunmiiii�����������!����������������������������HNU`n��������znUH?FH��������������������t���{������������qqt����������������������������������������������#��������������������������������������������������������������$)06BGOU[\[YPOB63)$$*0<AEIUYabb_[UI<30**�����

 
�������MN[gt�������tge[VMMx{��������������}yux��������	#������
#&%#
KNT[gnonlhg_[RNLIIKK#0>FGGIEA(#���������������������)1679;<<;6)��� #����������������������������������������������������������������������������������������������������,6>BO[ba[ZQOB96*,,,,Q[_hjt}}tkh[YPQQQQQQSUacfea`USSSSSSSSSSS5BI[g�����g[NB6525����������������������������������������������������������	
#).#
										��������������������"0<IU]bjieb\UI<20 "��������������������������������������������������������S[g������������|tg[S����������������������������������������-/<=BC<6/-----------rz���������������zor��	���������#)5<??<5) ��
�����������!(-0)������������������������agiqtw|��������tig^a������������������������������(�,�(�%���������¦ ¦««®¯§¦¦¦¦��������'�4�@�F�S�M�G�@�4�/�����������������������	������������������ֺԺɺǺɺֺ������׺ֺֺֺֺֺֺֺֿ����������������ĿʿͿοʿĿ��������������r�f�N�O�V�f�r�������������żѼҼ�����¿½µ¿������������������������������¿�	���	�	���!�"�'�"��	�	�	�	�	�	�	�	�M�A�4�3�2�<�A�M�Z�f�s�����������i�Z�M�T�O�I�S�T�]�`�m�y���������������~�m�`�T�t�g�[�N�:�5�)��)�5�B�N�[�g�t�t�"�	��ܾξǾʾ׾�������"�/�5�3�.�"ÜÔÏÜìù��������������������ùìÜ�U�R�P�U�U�a�n�u�z�|�z�u�n�a�U�U�U�U�U�U�ʼü��������ʼּڼ���ּռʼʼʼʼʼʾ������������������������˾Ծ׾ܾ׾Ͼʾ��4�,�.�4�M�Z�f�s��������˾Ͼ�����s�M�4���������j�A��3�N�d�������������������ϼ���t�p�m�l�r��������ּ�����ʼ����f�`�Z�M�H�M�Z�f�s������s�f�f�f�f�f�f�;�8�/�"��"�&�/�9�;�H�O�R�H�A�H�H�H�;�;����������������������
����
�������ݿݿѿ̿ǿοѿݿ������������ݿݿݿ�ƳƮƧƠƚƚƧƳ����������������������Ƴ��������������$�&�&�$��������������׾��������;׾�����	���!�"������w�s�v�w��������������������������*� �)�*�6�@�C�O�\�^�\�O�C�6�*�*�*�*�*�*�<�/�*�#��"�+�/�<�H�U�Y�Z�e�e�_�W�U�H�<àÞÙ×ÓÏÓàìóù����������ùìàà�˻ǻ����������������ûܼ��,�*�#�����A�8�/�2�<�N�Z�g�s�������������s�g�Z�N�A�e�Z�Y�L�L�R�Y�b�e�r�~�������������~�r�e�����{�z�~�y�|�����������ÿȿɿпտϿ���ƎƉƁ�yƁƎƚƠƧƨƪƧƞƚƎƎƎƎƎƎ�Z�T�V�c�v���������������������������g�Z�ֺκӺֺ�������������ֺֺֺֺֺ�������������������������������������������������սѽݽ�����(�4�C�H�A�<�(�D�D�D�D�D�D�D�D�D�D�D�EEEE	EED�D�D�����������	���!�����	���������#�=�M�f�������r�f�^�M�@�'������������������ּ���!�+�-� ����ּʼ����������������ɺ˺ͺɺź����������������������������������ĿſѿܿѿͿĿ���������ӻ»��ûлܻ�����4�>�O�M�@�4�'��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�T�K�E�D�M�U�Z�f�s�������������s�f�Z�-�,�-�5�:�F�S�\�^�S�F�:�-�-�-�-�-�-�-�-ĿĻıİķ����������������������������Ŀ�������������������*�6�C�6�+�*���ŠşŠţŭŹ������������ŹŭŠŠŠŠŠŠ�6�6�6�9�@�C�O�Z�[�O�K�C�6�6�6�6�6�6�6�6Ň�|ŇŋŔŠšŤŠŔŇŇŇŇŇŇŇŇŇŇ���������|������������������������������ù����������ùϹӹܹ��ܹϹùùùùù�������(�4�5�9�5�(����������	�����&�5�9�9�B�E�I�[�i�h�d�W�5�)��	�л̻û»ûƻлѻܻ��ܻлллллллл����{�r�q�z�������������̻λɻŻû»�����ܹܹӹܹ߹�������������������ݿܿۿڿݿ������ݿݿݿݿݿݿݿݿݿ��g�\�Z�T�Z�Z�g�s�~�������s�n�g�g�g�g�g�g������!�%�-�:�F�L�S�[�f�_�S�F�:�!��G�>�;�5�2�.�+�.�;�G�T�V�^�`�`�`�Z�T�G�G���׾ɾþþϾ׾������ �"�*�"�����L�A�L�R�[�h�r�~�����������������~�r�e�L���������������
�#�/�2�<�U�\�O�/�!������V�R�J�J�V�b�d�o�u�s�o�b�V�V�V�V�V�V�V�V�����������$�0�6�;�=�I�=�0�+�$����F1F*F0F1F=FJFTFJFCF=F1F1F1F1F1F1F1F1F1F1�Ϲ˹ùϹӹ�����!�'�.�-�'��	����ܹ��U�N�S�U�a�g�n�zÂ�z�x�n�b�a�U�U�U�U�U�U��������ùìàØßìù����������������轫���������Ľнݽݽ޽�ݽֽнĽ����������G�C�<�G�S�S�S�`�l�y�~�y�v�l�`�S�G�G�G�G�B�@�;�1�6�9�B�P�[�h�tāĉċĉ�{�[�O�I�B�����������������ĿƿĿ������������������0�%�#���
�������
��#�,�<�@�G�I�I�<�0������!�,�+�!����������� j X U K J W 7 M V , 3 n V d % : + ^ @ 0 g L / 3 c U \ , h M D W 2 6 i C x = Q \ n P d W 3 L N F - > N i J L [ H 9 j M ;  - l & > : . O J = Q ` � s o � 7 Z � ` ;    0    �  �  =  M  �  �  V  �  �  �  �    �  z  _  9  �  �  �  �  c  �  �  �    &  �  �  $  +  \  ,  �  w  �  �  �     $    _  5  H  3  �  �  �  �  x  I  �  |    	  �  f  �  A  A  ^  Y  �  �  �  Q  y  =  |  7  3  �  �    �  �  *  �  W  ><49X;D���t�$�  ;��
���
�H�9��j�o�����T����1��1�ě����
���㼼j�+����Ƨ�1��9X�+�+�o���<j��w���}�D�����T����#�
�@��+�u��w�,1�Y��8Q�@��@���O߽#�
�u��7L�]/�}�<j���T�49X�H�9�8Q�,1�L�ͽ]/�<j��+�P�`��Q�@��@��aG����
�}�ȴ9�ě���Q�y�#��o�y�#�}󶽋C�������C�����ě���O߽�����E�B�BA�BbNB?!BA:B@�B kB�BB��B�B��BWsB0�A�EB!B&�}B4�|B
�B'W}B�$B"�:B�B,DB ��A���B �B��BTiB3�BWBۮBwB��B"��B,q�BN�B�FB�VB4�B&��B3)B	�B)�B-p�B$��B��B%;+B��B�8B]�BjB\;Bm�B�Bs�Bf!B�B��B��B�B<B!5|B`BVYB&�)B�-B+HB�@B
��B��B�QB�6B�JB4fBQ-B5B��B��BPKB	�3B"�sB�B;YB?�B��B?�B>�B��BA�B��B�B��B�B0C�A��B!�rB&B4�B��B'��BGKB"��B�1BKB �OA�o�B �BB?B>�BG
B��B�B�B"A�B,EiBB�B�}B�B��B&�pBycB	�yB)�DB-�VB$�uB�B%@�BU�B�lB��B��B>2B��B�2B�}B?�B��BG�BA�B�pB@B!�B�_BH
B&�<B�B?BB=WB
�$B«B��B��BC�B?B��BA,BABB��BqcB	�,B"ιA1�YA�N�@Ȇ-A�p�@@IbAv�@��WA�VhA�+A?��Al:A�F�AY��Aͥ�Aƾu@�2�AM�ZAE�A��@�U=AA��A�d�A��A}��B�B��AW��AGzB �@A��iA���@�	�A��?�Au0�B�UA�2�@H
�A��A2�qC�A*AY��@ե"Ajj@,�0Au�@���C�9AC�@�.�A���A��aA���B �A�n@��H>ps�A��A�{�@�Q�@�[2?�BA}�	A�ȕ@wN�Aem�AWLM@	�A���B%5B	�zC��?=��A��OA�2FA'NAn�A�H+AvtA�x�@f|�A0��A���@�:1A�s�@<�Av��@���A�yA��iA=(Am[�A�|oAZ�@A��YAƲ�@��AM&fAB؆A��R@�ߤAB��A�3�A�!�A~�rB��B��AX�dAG�B �UA�~�A̫�@�Z4A��&?��Ar�B��A��A@Ev�A��rA4�C�LhAZ��@��'A�@+�vAt�@�P�C�8�AB��@��#A�)A�Y.A�xB5�A�h@�+�>F_aA��A�M�@�l�@�(�?��A~��A�wM@tAe_�AV�4@�A���BB	��C��w?KW�A� �A΄`A)�WA4A�SAv��A��@d`            
         9                           	         5   T               
               %      6   0                                 $                   
   ,      
                        /         
         0   -   &   	      	   
            
          	                        #                  #               %   =   '                                    '         )      -                  !   1         %                                    #                           !   %                                                         #                                 #   5   %                                    #         )      -                  !   )                                                                           %                                    NۖN�O)�?N'Q�N�6N�2�O���N�{lN3��OgiO7�O7�iOA�-O4�wNd�>Nd%�N�YO�MmPy�`O�\Nk@�N�+6N�2?N���O"`~N�^�O[�N�,�N{�hN���N���O�I�O��aO��P�-Nbh�P �N�f�N��Oc��N���O(�O�YO��N'�WOI O@gNg0LOu�Nl�FO��!N첺N�b�NV?M�K�N��oNrg�N>\O��N+dGO!��NB�!N�xN�}�OD>N�HvO�O��Oө�N�kO� N	��O3�N\��OONKG�N�0�O"lNWXOv�N$��      *  �  8  �  �  �  �  �  �  A  {  ~  I  �    6    	�  q    �  �  P  T  �  �  �  &  J  %  �  a  f  �  
  �  �  �  �  �  �  �  �    �  .  �    	�  Q  �  �  �  �  v  �  X    �  �  �  �  �  �  p  >  �  �  �  �  J  �  �  ?  j  �  *  =  �<�j<T��<t�;ě�<t�<o��o���
:�o�D���D����o�o�#�
�49X�49X�u�u��C������u����ě���9X��j��j��h��h�����@��+�\)�C���h��h���C��o�o�o�+�\)�C��#�
�\)�t��<j����P��P���#�
�#�
�#�
�#�
�#�
�,1�,1�8Q�49X�q���49X�8Q�<j�]/�T���ixսm�h�]/�Y��Y��Y��Y��e`B�}󶽁%��o��\)��+��7L������������������������������� 
��������)6=@BDDB:62)& ��������������������
	���#)��������z�����������������|z#(/<HHUVUUH</#��������������������"$.<HUX\[TPLH</-'#!"46BOO[]dhc_[OB<:6/-4�� #'*+,++/.#�"#'.6COVX\]\WPD6*`amxz�������zwqma[Z`��������������������9<GIUWaYUKIF=<999999��������������������|���������������zww|#0Sn{������{U<0�����	����������������������������lnrtz��������znnnll[[]cht�����thd[[[[[yz������������zpyyyyRT[amqywrmlija]TMLORimz���������zunmiiii�����	�������� ����������������������������`acnz|�����zna``````��������������������uz���������������tsu����������������������������������������������#���������������������������������������������������������������$)06BGOU[\[YPOB63)$$*0<AEIUYabb_[UI<30**�����

 
�������P[gt}�������tgg[WOPPx{��������������}yux������� �������
#&%#
KNT[gnonlhg_[RNLIIKK#029<:<;70%#
���������������������)1679;<<;6)��� #����������������������������������������������������������������������������������������������������,6>BO[ba[ZQOB96*,,,,S[htzvthhg[QSSSSSSSSSUacfea`USSSSSSSSSSS5BGNX[gtz�|g[NB?7625������������������������������������������������������������	
#).#
										�������������������� #*0<IXbgfbUI<630&" ����������������������������������������������������������T[t������������}tg\T����������������������������������������-/<=BC<6/-----------rz���������������zor��	���������#)5<??<5) ��
�����������%+,)������������������������agiqtw|��������tig^a������������������������������(�,�(�%���������¦ ¦««®¯§¦¦¦¦���������'�4�@�E�M�Q�M�F�@�4�.������������������ ���������������������ֺԺɺǺɺֺ������׺ֺֺֺֺֺֺֺֿ����������������ĿʿͿοʿĿ����������������r�c�Y�X�b�f�r������������̼̼�����¿º¿������������������������������¿¿�	���	�	���!�"�'�"��	�	�	�	�	�	�	�	�M�A�4�3�3�=�A�M�Z�f�����������s�f�Z�M�T�O�I�S�T�]�`�m�y���������������~�m�`�T�t�g�[�N�:�5�)��)�5�B�N�[�g�t�t�"��	�����ӾϾ׾�����	��"�)�.�$�"àÞÖÓÓàìù������������������ùìà�U�T�Q�U�W�a�n�t�z�{�z�p�n�a�U�U�U�U�U�U�ʼü��������ʼּڼ���ּռʼʼʼʼʼʾ��������������ʾ̾ҾʾȾ����������������A�8�4�A�C�M�Z�f�s���������ǾǾʾ�����A���n�a�A�.�/�G�j����������������������������s�q�p�p�w����������мۼ��ּ������f�`�Z�M�H�M�Z�f�s������s�f�f�f�f�f�f�;�8�/�"��"�&�/�9�;�H�O�R�H�A�H�H�H�;�;���������������������
����
�����������ݿݿѿ̿ǿοѿݿ������������ݿݿݿ�ƳƮƧƠƚƚƧƳ����������������������Ƴ��������������$�&�&�$�����������������׾оǾʾ׾ܾ����	������	��������{�|������������������������������*� �)�*�6�@�C�O�\�^�\�O�C�6�*�*�*�*�*�*�U�O�H�A�<�<�7�<�H�U�U�Y�Y�U�U�U�U�U�U�UìäàÝÜàìù��������üùìììììì��л̻û����������ûܻ���(�'� �����A�;�5�2�5�@�N�Z�g�s���������~�s�g�Z�N�A�e�Z�Y�L�L�R�Y�b�e�r�~�������������~�r�e�����{�z�~�y�|�����������ÿȿɿпտϿ���ƎƉƁ�yƁƎƚƠƧƨƪƧƞƚƎƎƎƎƎƎ�g�\�Y�d�m�x���������������������������g�ֺκӺֺ�������������ֺֺֺֺֺ�������������������������������������������������սѽݽ�����(�4�C�H�A�<�(�D�D�D�D�D�D�D�D�D�D�D�EEEE	EED�D�D����������	��������	����������#�=�M�f�������r�f�^�M�@�'��������������ʼ�����%�'������ּ������������������ɺ˺ͺɺź����������������������������������ĿſѿܿѿͿĿ������������ܻλλлܻ������"�'�)�,�'��E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�T�K�E�D�M�U�Z�f�s�������������s�f�Z�-�,�-�5�:�F�S�\�^�S�F�:�-�-�-�-�-�-�-�-ĿĻıİķ����������������������������Ŀ�������������������*�6�C�6�+�*���ŠşŠţŭŹ������������ŹŭŠŠŠŠŠŠ�6�6�6�9�@�C�O�Z�[�O�K�C�6�6�6�6�6�6�6�6Ň�|ŇŋŔŠšŤŠŔŇŇŇŇŇŇŇŇŇŇ���������|������������������������������ù������ùϹйܹܹܹܹϹùùùùùùù�������(�4�5�9�5�(�����������������(�1�5�B�[�g�g�c�V�N�5�)���л̻û»ûƻлѻܻ��ܻлллллллл������������������������»ûŻû���������ܹܹӹܹ߹�������������������ݿܿۿڿݿ������ݿݿݿݿݿݿݿݿݿ��g�\�Z�T�Z�Z�g�s�~�������s�n�g�g�g�g�g�g�!������!�'�-�:�F�U�`�[�S�F�B�:�-�!�G�A�;�6�4�;�;�G�T�U�]�_�X�T�G�G�G�G�G�G��׾;ƾǾʾԾ׾���������	�����r�j�e�Y�T�W�a�e�l�~�����������������~�r���������������
�#�/�<�J�U�[�N�/� ������V�R�J�J�V�b�d�o�u�s�o�b�V�V�V�V�V�V�V�V�����������$�0�6�;�=�I�=�0�+�$����F1F*F0F1F=FJFTFJFCF=F1F1F1F1F1F1F1F1F1F1�Ϲ˹ùϹӹ�����!�'�.�-�'��	����ܹ��U�N�S�U�a�g�n�zÂ�z�x�n�b�a�U�U�U�U�U�U��������ùìàØßìù����������������轫���������Ľнݽݽ޽�ݽֽнĽ����������G�C�<�G�S�S�S�`�l�y�~�y�v�l�`�S�G�G�G�G�O�K�B�>�B�O�U�[�h�tāąĈĆā�x�h�e�[�O�����������������ĿƿĿ������������������0�%�#���
�������
��#�,�<�@�G�I�I�<�0������!�,�+�!����������� j X T \ J W 1 Q V * 3 n L ? ) :  [ - / g L - 3 c U W ) h k / ^ % 6 i C s = Q \ n M d Y 3 L > D - > N i J L [ H : j O ;  - l & < , " B I = Q ` � s o � 7 Q � ` ;    0    �  T  =  M  _  O  V  �  �  �  �  �  v  z  �  "    ?  �  �  �  �  �  �  w  �  �  �  �  �    ,  �  w    �  �     $  ]  _  :  H  3  �  j  �  �  x  I  �  |    	  �  f  W  A  U  ^  Y  �  �  �  �    &  |  7  3  �  �    �  �  y  �  W  >  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7  F7        �  �  �  �  �  s  U  ?  /    �  �  \    �  �        �  �  �  �  �  �  �  �  �  f  L  3    �  �  X  �  ~  !  *         �  �  �  �  �  x  c  H    �  �  {  1  �  #  S  a  n  y  �  �  �  �  �  x  R  4    
  �  �  �  �  �    8  +        �  �  �  �  �  �  �  �  ~  o  _  Q  B  3  $  �  �  �  �  �  �  �  �  �  |  f  H  #  �  �  ?  �  H   �   B  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X  �  T  �  �  �  �  %  `  �  �  �  �  �    c  @    �  �  u  7  #  �  _  �  �  �  �  �  �  �  �  �  �  �  �  {  n  a  R  C  4  &      �  �  �  �  p  d  Y  K  D  ,    �  �  k    �  �    y   �  �  �  �  �  �  �  y  g  T  <  #    �  �  �  {  ]  @  2  &  A  4  /  (  "      �  �  �  �  �  i  B    �  E  �  �  �  H  K  X  m  z  w  r  m  e  _  [  T  E  -    �  �  J   �   L  X  ?  i  n  T  8    �  �  �  �  ]  ;    �  �  �  �  P    :  B  I  I  H  E  B  =  5  (  !        �  �  �  b  A  "  �  �  �  �  �  �  �  �  q  T  2    �  �  �  r  I      �   �  
                      �  �  �  �  ~  R     �   �  "  3  6  0  (         �  �  �  �  �  �  �  ~  P  &  �  �    
    �  �  �  �  �  �  �  \  4    �  q    �  7  �    	�  	�  	�  	�  	�  	�  	�  	�  	}  	c  	6  �  �  �  $  l  �      �  q  a  Q  A  3  &      �  �  �  �  �  �  �  �  �  �  �  �                     #  %  (  *  *  #            �  �  �  �  �  �  �  �  �  �  �  �  i  M  .    �  �  �  \  �  �  �  �  �  �  �  u  ^  C    �  �  �  �  X  (  �  �  �  P  K  G  >  5  )      �  �  �  �  �  �  n  O  /    �  �  T  N  I  B  7  ,      �  �  �  �  e  @    �  �  �  �  e  V  �  �  �  �  �  �  �    ^  2  �  �  k    �  �  p  8  �  �  �  �  �  �  �  �  �  �  �  �  �  \    �  �  X    �  y  �  �  �  �  �  z  p  g  Z  I  7  %    �  �  �  �  �  �  �  �  �  �  z  $  �  :  l  �  �    &    �  �  �    �  �    6  7  8  7  4  ?  H  >  .    �  �  �  W    �  }  ,  �  Z  �       #       �  �  G  �  �  V  P    �  \  1  �  �  �  �  �  �  �  �  �  �  m  ?  
  �  �  1  �  @  �    l  �  �  a  Q  A  /        �  �  �  �  �  �  k  S  <  *        f  X  I  :  '    �  �  �  �  u  H  !    �  �  q  >  ,   �  �  �  �  �  �  �  �  �  �  �  �  p  a  Q  A  2  "       �    	    �  �  �  �  �  �  �  �    �  �  �  `  	  �    F  �  �  �  �  u  b  P  ?  .      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  r  p  l  g  U  A  %    �  |  o  ]  H  3      �  �  �  �  �  d  +  �  �  5  �  1  �  �  �  �  n  S  G  )    �  �  q  [  J  <  !    �  �  �  �  �  �  �  �  �  �  �  �  �  u  T  .    �  �  o  2  �  ;  �  �  �  �  �  y  i  ^  N  =  *    �  �  �  �  �  n  T  0  �  �  �  �  �  �  �  �  �  ]  +  �  �  r  A  �  �  $  {   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  r  k  c      �  �  �  �  �  �  m  <    �  �  B  �  �  b    �  �  i  v  v  s  k  r  �  ~  k  S  5    �  �  �  e  $  �  I  �  �  &  #      �  �  �  �  �  �  `  >    �  �  �  �  j  !  �  �  �  �  �  �  �  �  �  s  X  4    �  �  -  �  M  �   �      �  �  �  �  �  �  y  ]  ?    �  �  |  =    �  �  �  	�  	�  	�  	O  	  �  `  �  �  X    �  �  C  �  C  �    �  |  Q  L  G  C  >  7  *        �  �  �  �  �  �  �  l  W  C  �  �  �  �  �  �  �  �  x  b  L  1    �  �  h  #  �  �  P  �  �  �  �  �  ~  v  n  f  _  X  Q  O  Q  T  V  m  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    x  q  j  c  \  U  �  z  d  I  .    �  �  �  �  �  �    g  O  8    �  �  }  F  ^  n  t  l  b  W  K  >  2  %    	  �  �  �  �  |  !  �  �  �  �  �  �  �  �  �  �  �  n  S  8      �  �  �  �  j    W  N  A  6  (      �  �  �  �  �  �  q     �  R  �  $          �  �  �  �  c  /  �  �  �  S    �  �  L     �  %  G  f    �  �  �  �  �  n  L  '  �  �      e  �  �  {  �  }  v  n  g  `  X  P  F  =  4  *  !        �  �  �  �  �  �  �  u  g  Z  M  ?  2  %      �  �  �  �  �  �  �  �  �  �  ~  p  c  V  J  <  -          +  F  a    �  �  �  �  �  �  �  �  e  ?      �  �  �  `  &  �  �  '  �    h  m  �  �  }  l  [  I  5      �  �  �  �  ^  8    �  �  �  )  ]  m  m  a  P  6    �  �  \    �  m  �  q  �    @    �    =  <  3  *      �  �  �  �  W    �  ,  ~  �  s   �  �  �  �  �  �  �  �  �  k  H    �  r    �     [  �  U  �  �  g  O  5       �  �  �  �  t  V  8    �  �  �  �  l  B  �  �  �  �  �  �  �  �  �  }  p  c  V  B  +    �  �  <  �  �  �  �  n  V  ?  )    �  �  �  �  �  �  {  L    �  �  �  J  D  =  *      �  �  �  �  |  Y  2  	  �  �  a     �   �  �  �  r  I  $    �  �  �  �  �  e  1  �  �  �  O    �  �  �  �  r  >      2    �  �  �  �  �  �  l  E    �  �  �  ?  A  C  E  E  ?  9  4  ,  "      �  �  �  u  Y  ?  &    j  e  a  W  M  ?  1  "      �  �  �  �  �  �  �  t  k  b  W  �  �  �  �  �  �  g  >  �  �  L  �  l    �  �  '  L  f  *  !        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  =  &    �  �  �  �  �  f  J  3  "    ,  C  !  �  �  N   �  �  �  �  �  �  x  o  g  ^  U  K  B  ;  4  E  u  �  �  �  �