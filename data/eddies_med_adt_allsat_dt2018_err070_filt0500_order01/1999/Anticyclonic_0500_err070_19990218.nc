CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�$�/��        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�~�   max       P��$        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �0 �   max       >��        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @Fq��R     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @vv�\(��     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�'�            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >.{        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��	   max       B1}�        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�um   max       B0�*        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?@ȇ   max       C���        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?0o   max       C���        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          #        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�~�   max       O߾f        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�S��Mj   max       ?����C�]        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �0 �   max       >��        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @Fg�z�H     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vv�\(��     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F   max         F        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?t��n   max       ?��C,�{        W�         	         $   �   
            ]      
            E                           *                     �      1   I         '   4   B      -            *   "   
   *   ]   	            	   
            %      N_�N%؃N��,NQ5�O��O}V�PAH�OwD�M�~�O!7O`,�P��$Ng�N�]�M��lN�D-O?:Pb =NNI�Oe�kO��&Nǆ�NjZ)NIfO8�$O��P�Nb�N�%HN.:NJ�gN�LcN��hO���NLl�O�B�O���N7��Og�O�A=OȁKO�� N}O�")N�`�O�O(51O�d7O��aND�POܺP#[�N���OlQN��N���N�t�N'ESOF�N��O	��O�~�N���NES��0 ż�h��C��49X�o���
�o:�o;o;��
;ě�;�`B;�`B;�`B<t�<t�<49X<49X<D��<D��<T��<e`B<e`B<�o<�C�<���<���<�9X<�j<�j<���<���<���<�/<�/<�`B<�`B<�<�<��<��=+=�P=�w=#�
=,1=,1=0 �=0 �=0 �=8Q�=H�9=H�9=Y�=]/=]/=aG�=ix�=�o=�\)=��=��
=�Q�>��0.03:<?HIJKI<0000000X[gtvvtg\[XXXXXXXXXXO[ggt��������ytg^[OO��������������������vuwz���������|zvvvv��������������������"5N[gt�����tgN5'OKLS[ht���uvwphc`[VOPRTU[becbUPPPPPPPPPP��������������������0*-135BDNR`imjg^[NB0NQalq������������mTN./1<>GHJIH<93/......!)6BEDGDB62)\hkt����th\\\\\\\\\\|v����������������||
)5=BA95)������"&
��������HHLHEHRTX]\UTSIHHHHH��������������������������
����� "(/0;DHTWTMH;7/,(" ABOS[dhokhe^[OKBAAAA_Xantz}zna__________emnrz�����������tne������
%'..0#
����	�
#/>FGKC</#	��������������������njlnwz}��������zznnnBBDJOZ[^[[OBBBBBBBBB416<HOSJH<4444444444zz���������������zzzgknw{���������{ngggg�������

�����,-/<HINH=<</,,,,,,,,������),/440���,*3=><BO[hx~~wh[JB),@BO[\][[OJCB@@@@@@@@-+.6O\hu{���yuh\OC6-��
#/<BGHB<:/#
��#/7CUryxtnaUH</%�������

������������������������}xx|���������������}
*0566;64*!

�����	
��������YOO[gjt���������tg]Y���������� � ���������)6BFC6)�����������������������babhrt����������tmhbamz�������������mgfapt}������������yvtpp80;?HTabmonmfaTH?;88������������������!#/<AHKQKH<5/+#!!!!%)*-58BCDBB95)+)%%%%��������������������"&/;HNSNF;/*"xwy{|��������{xxxxxx )58=85)( ������(!��������������������������������������������������������������������������������ŇŏŒŌŇ�{�w�t�{�|ŇŇŇŇŇŇŇŇŇŇ�������������������������������������������#�*�6�8�6�0�*�%���������������������
�	�������������������������ٻ�����'�0�:�>�<�4�'�����ܻлɻۻ��)�6�B�[�r�r�m�d�O�A�6��������������)���ʾ׾�����׾ʾ�����s�l�s���������@�M�Y�_�Y�R�M�@�>�7�@�@�@�@�@�@�@�@�@�@�tāčĚĜĦįĳĸķĳĦğĚčā�t�k�j�t����*�6�@�:�6�/�*�����������������A�Z�q�������������������s�Z��� �6�8�AEiErEuEzEuEiEdE\EPEGECEPE\EfEiEiEiEiEiEiÓÙàããäàÓÇ�z�y�t�zÀÇÇÓÓÓÓùþ��������ùøò÷ùùùùùùùùùù�#�/�;�<�G�C�<�:�1�/�)�#�������#�#�/�<�H�U�a�n�q�p�l�g�a�U�H�<�7�/�0�/�(�/�[¦¿��²¦�[�5���������������)�[��������	��	�������������������������(�A�M�Y�a�b�a�X�[�Z�M�A�>�4�(�#����(�������ʼ̼˼ʼ���������������r�q�y�����H�P�H�E�H�M�H�E�@�;�5�"���"�,�/�;�E�H�H�I�O�U�]�a�k�a�U�K�H�<�:�9�<�G�H�H�H�H�/�<�D�>�<�4�/�)�&�.�/�/�/�/�/�/�/�/�/�/E�E�E�E�FFFFFFE�E�E�E�E�E�E�E�E�E��	�"�/�;�F�I�B�3�"��	�����������������	�������������������������Z�J�8�/�5�N�g�������������½������������|��������������������������������������������������޽y�{�������������{�y�m�x�y�y�y�y�y�y�y�y�n�zÇÉÎÇ�z�n�k�k�n�n�n�n�n�n�n�n�n�n�
�����
�����������������������
�
��������!�$�!������������������DbDoD�D�D�D�D�D�D�D�D�D�D{DoDZDHDHDKDVDb¥�_�l�������������t�_�S�F�:�-���!�*�F�_�M�Y�r�����ʼڼڼҼǼ������r�[�S�M�E�>�M�4�7�;�5�4�'�����'�+�4�4�4�4�4�4�4�4�G�T�`�i�o�d�]�Z�T�@�;�7�.�,�&�&�)�0�=�G�y�������������������y�`�U�T�G�J�T�c�m�y�	��"�,�:�.�"������׾ξþ����ľؾ�	������������ɺ�����������ɺֺ⽅���������������������������������������(�4�A�M�Z��������f�M�4��������(�y���������������y�x�l�b�`�[�`�d�l�u�y�y�"�.�;�@�D�<�;�.�"���	��� �	��"�"�"�"�(�5�A�M�M�N�S�P�J�A�5�(����!�)�+�(�(�������1�3�<�3�'�������׹̹ƹչ�Y�r�������������������~�k�^�Y�W�N�L�B�Y�F�R�S�S�_�l�s�l�j�_�S�I�F�;�F�F�F�F�F�F�@�L�Y�e�r�w�~�r�q�e�Y�L�D�@�3�3�1�3�6�@������������������ìÓÁ�v�wÁÃÒàì�ƻ-�0�-�+�-�7�-�%�!���� ����!�$�-�-ĚĦĮĳĸĴĳĳĩĦĚėčĉĆĊčēĚĚ�����(�$����������������������������������������������
�����
��������������������������������������~�{����������������������#�0�9�<�B�G�D�<�0�#�����������
���#�������������������������~�}�������������m�y���}�y�x�p�c�`�T�G�F�G�G�K�Q�T�`�b�m�T�a�n�q�o�h�^�Q�N�H�;�/�'�����/�;�T�#�0�<�I�U�a�`�U�I�<�3�0�#��#�#�#�#�#�#�!�.�2�5�.�!����������� �!�!�!�!�!�! : 7 R M 8 G  j G Z ; 1 a 8 � ? # j v Q & S p a ) 1 T ~ Q 8 4 o c 3 2 C O < V  J ` , 1 4 * ` : D M ) 0 C 4 + U H D < < [ 5 t |  q  L  I  v  (    "  a    r  �  �  }  �  x    �  �  �      �  �  @  �  y  �  �  �  J  d  �  �  �  d  �  s  _     0      -  B  �    �  *    S  V  �  �  7  �  �  �  V  �  �  Z  z  �  ��#�
��/�o�o;D��<�>.{<#�
;��
<���<ě�=���<��
<�o<T��<�9X='�=���<�o<ě�=o<�1<��
<ě�=t�=@�=y�#<���=C�=C�<�<�=�w>.{=�w=���=��=t�=49X=�\)=���=ȴ9=�w=�1=D��=m�h=aG�=�{=��-=Y�=� �>bN=ix�=�C�=�o=��P=�o=�+=�{=�1=�E�==Ƨ�>-VB&E�B	?�B	âB��B n�B!ɛBh�B��B'jB��B(�BՍBƣB��B	/B�TBu�B�A�r�B!��B#6,A��rB
RB��B��BQ�B�uB�GBc,B�kB�nBXB(�/B	+B��B�B44B��B1}�BİBJB#��B-�B�kB/y5B(HB	�?B�OB��B b]B�B��B:�A�OB�&B��BC�B�DA��	B)O>BNWB�B�+B?�B&K~B	\B
�B�[B B�B!��B@AB>�B'?B�[B@'B�dB�B�2B?�B��B�B�WA���B!(ZB#@RA��"BGFBD|B@B@@B��B��BEzB�JB.�B��B(��B2+B¸B��B?�B�$B0�*B:;B?�B#K�B,�<B��B/�BB?dB
�B��B��B @�B~�B9bBBBA��"B��B��B?�B�UA�umB)@�BN�B��B��B@@�y.A�2�A���A�ŝBG@�o2A֭�ALߴ@�C�A���A�D�A�f�C�̹A�EA͢�A�p�A�
tA��!A��dA:�@�7zA�)A�[�A���C���A���A�zA �}A�L�Aw�Aȅ�A��@^\C��_A��@���@�|7@��eAe��Al�)AX @=��A/�A9��A�A^��A�o�?@ȇ@Z�@���?�K�A�~@f]TA��eAԩ'A�/HA�rCAG�A�E�A �AiyA�mA쳭A��@���A�A��A�z%B;7@�JA֦kAPE�@�-�A�N�A��~A�~�C�ԋA�!2A��ZA�~�A�TrA��{A�Y^A:��@��A�@A�s#A�~�C���A��A��_A#כA��LAU�AȐA�v!@WgC��0A��@�I�@��H@�n�Ag Am�AV��@K��A�A8�!A	�A^�A�|�?0o@�@�	�?ԥA̢�@b��A߅�Aԁ�A���A�W�AGU�A�KA ��Ai �A�� A�xqA T         
         $   �   
            ^      
            E                  	         +                     �      1   J         (   5   B      .   	         +   #      *   ]   	            
   
            &                           +               9                  ;                           )                            !   )            #   %      #                        *                                                                        #                                                                                                                                                            N_�N%؃N��,NQ5�O��O��O���OwD�M�~�O!7O:� O߾fN@X�N��M��lN�}O5�1O�LNNI�O3&�Oc!N��NjZ)NIfO*�OIەO��Nb�N�աN.:NJ�gN�LcN��hO1c�N5j�OQ��O_��N7��OO�O�A=O�܍O�U�N}O:�N�`�O�O(51O%�Oh�;ND�PN���O�Q?NXq�OlQN��N���N�t�N'ESOF�N��O	��O�;�N���NES�  �  -  �  n  H      �  �  I  5  �  b  R    �  }  �  �  �  k  �  �  �  �  o  r  �  �  �    �  V  �  �  �  	�     �  �  �  �  �  �  *  <  >  �    |  �  G  �  �  g  (  �  V  �    �  d  �  /�0 ż�h��C��49X�o;��
=�+:�o;o;��
<o=@�<o<t�<t�<#�
<D��=aG�<D��<e`B<u<u<e`B<�o<�t�<���=+<�9X<���<�j<���<���<���=�E�<�`B=49X=T��<�<��<��=#�
=<j=�P=u=#�
=,1=,1=aG�=L��=0 �=ix�=���=P�`=Y�=]/=]/=aG�=ix�=�o=�\)=��=��=�Q�>��0.03:<?HIJKI<0000000X[gtvvtg\[XXXXXXXXXXO[ggt��������ytg^[OO��������������������vuwz���������|zvvvv��������������������(+3BN[gqxzyupj[NB1*(OKLS[ht���uvwphc`[VOPRTU[becbUPPPPPPPPPP��������������������2-045BNP]gjig[NIB;52��������������������//4<=FHJHH<94///////!%)6>BBEB:65)!!!!!!\hkt����th\\\\\\\\\\y�����������������yy	)5<A@85)�������	


������HHLHEHRTX]\UTSIHHHHH����������������������������������+/3;BHTUTPKH;9//++++ABOS[dhokhe^[OKBAAAA_Xantz}zna__________knotz������������vnk�������

����#/<=@A@<8/#��������������������lmnz�������zrnllllllBBDJOZ[^[[OBBBBBBBBB416<HOSJH<4444444444zz���������������zzzgknw{���������{ngggg�������

�����,./<HHMH</,,,,,,,,,,�����$&%!��@ABEFHO[hotwumh[OIB@@BO[\][[OJCB@@@@@@@@.-06CO\hux|�uh\OC:6.��
#/<BGHB<:/#
��)&$#*/8<HUkssmaUH</)������

���������������������������������������������
*0566;64*!

�����	
��������YOO[gjt���������tg]Y������������������������)68>>>76)���������������������gghnt��������tihgggg�||}����������������}}������������}}}}}}80;?HTabmonmfaTH?;88������������������!#/<AHKQKH<5/+#!!!!%)*-58BCDBB95)+)%%%%��������������������"&/;HNSNF;/*"xwy{|��������{xxxxxx )58=85)( ������"���������������������������������������������������������������������������������ŇŏŒŌŇ�{�w�t�{�|ŇŇŇŇŇŇŇŇŇŇ�������������������������������������������#�*�6�8�6�0�*�%���������������������
�	�������������������������ٻ������$�/�'�&��������ܻڻܻ���)�6�B�S�[�]�X�O�B�6�)�������� ���)���ʾ׾�����׾ʾ�����s�l�s���������@�M�Y�_�Y�R�M�@�>�7�@�@�@�@�@�@�@�@�@�@�tāčĚĜĦįĳĸķĳĦğĚčā�t�k�j�t����*�/�4�,�*������������������ ��N�Z�g�s�����������������s�g�N�<�3�3�A�NEiEmEuEyEuEiEaE\EPEPEFEPE\EgEiEiEiEiEiEiÇÓàááàÓÑÇ�~�z�y�zÂÇÇÇÇÇÇùþ��������ùøò÷ùùùùùùùùùù�/�8�<�E�B�<�9�0�/�+�#������#�#�/�/�/�<�H�U�a�n�p�p�l�g�a�U�H�<�8�0�0�/�.�/�B�N�[�g�n�g�c�[�N�H�B�>�5�)���)�5�:�B��������	��	�������������������������(�4�A�M�V�]�^�]�Z�R�M�A�=�4�,�(�!��%�(�������Ƽʼʼʼȼ�����������t�t�}�������;�@�H�K�H�D�<�;�:�/�"�� �"�/�0�;�;�;�;�H�I�O�U�]�a�k�a�U�K�H�<�:�9�<�G�H�H�H�H�/�<�D�>�<�4�/�)�&�.�/�/�/�/�/�/�/�/�/�/E�E�E�E�FFFFFFE�E�E�E�E�E�E�E�E�E��	��"�/�;�?�A�;�7�/�)�"��	�����������	�g�s�������������������������g�Z�M�P�Z�g�����������½������������|�����������������������������������������������������y�{�������������{�y�m�x�y�y�y�y�y�y�y�y�n�zÇÉÎÇ�z�n�k�k�n�n�n�n�n�n�n�n�n�n�
�����
�����������������������
�
��������!�$�!������������������D{D�D�D�D�D�D�D�D�D�D�D�D{DtDoDdDeDoDpD{¤�F�_�l�x�~���{�x�l�_�S�F�:�4�-�+�,�-�;�F�r������������ü����������r�k�d�c�f�m�r�4�7�;�5�4�'�����'�+�4�4�4�4�4�4�4�4�G�T�`�g�k�m�b�[�W�T�G�;�.�)�(�+�2�;�>�G�y�������������������y�`�U�T�G�J�T�c�m�y����	��"�'�"�!��
�����׾˾žþ;��������� ���ֺɺ����������ɺֺ�ｅ���������������������������������������4�A�F�M�Z�[�a�Z�W�M�A�4�$������(�4�y���������������y�x�l�b�`�[�`�d�l�u�y�y�"�.�;�@�D�<�;�.�"���	��� �	��"�"�"�"�(�5�A�M�M�N�S�P�J�A�5�(����!�)�+�(�(��������"���������ֹܹԹܹ��e�r�~�������������������~�r�p�d�`�]�Z�e�F�R�S�S�_�l�s�l�j�_�S�I�F�;�F�F�F�F�F�F�L�Y�Z�e�l�o�e�d�Y�O�L�@�:�8�@�D�L�L�L�LÓàìù������������������ìàÍÃÄÍÓ��!�%�-�/�-�!� ��	����������ĚĦĮĳĸĴĳĳĩĦĚėčĉĆĊčēĚĚ�����(�$����������������������������������������������
�����
��������������������������������������~�{����������������������#�0�9�<�B�G�D�<�0�#�����������
���#�������������������������~�}�������������m�y���}�y�x�p�c�`�T�G�F�G�G�K�Q�T�`�b�m�T�e�m�n�m�f�\�O�H�;�/�)����� �/�;�T�#�0�<�I�U�a�`�U�I�<�3�0�#��#�#�#�#�#�#�!�.�2�5�.�!����������� �!�!�!�!�!�! : 7 R M 8 9  j G Z 6 2 Y 2 � D " B v = % - p a %  P ~ E 8 4 o c / 4 ' 4 < T  E U , $ 4 * `   5 M # * & 4 + U H D < < [ 2 t |  q  L  I  v  (  K  �  a    r  �  �  _  �  x     �  6  �  �  �  �  �  @  t  �    �  �  J  d  �  �  w  W  �  �  _  �  0  `  3  -  �  �    �  \  �  S  �  �  e  7  �  �  �  V  �  �  Z  R  �  �  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  �  �  �  �  �  �  �  �  �  �  �  �  v  k  _  S  F  :  .  !  -  &             �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  n  \  I  4      �  �  �  �  �  u  b  O  :  %  n  `  R  D  6  )        �  �  �  �  �  �  �  �  �  �  �  H  G  E  A  ;  3  !    �  �  �  �  �  n  N    �  �  >   �  �  �  �                  �  �  �  <  �  $  �  �  �  �    �  �  4  �  �      �  �    u  �  �  c  �  

      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  D     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  I  5      �  �  �  �  c  ;    �  �  �  W    �  �  �  �    *  4  5  *    
  �  �  �  �  �  �  c  6  �  �  t  3  Z  �  /  �  4  �  �  �  �  �  �  �  �  �  M  �  r  �  P  �  <  H  Z  ]  S  =    �  �  �  �  P     �  �  �  U     �  �  �  P  M  J  M  Q  N  I  ;  (    �  �  �  �  y  S  *     �  �      *  8  E  O  Q  S  T  V  W  X  Z  [  \  ]  ]  ^  ^  _  �  �  �  �  �  �  g  G  )    �  �  �  [    �  y  '   �   ~  d  |  n  Z  @  &    �  �  �  `  $  �  Z  �  %  �    �  {  V  �  �  &  R  k  _  ;      K  ^  �  �  J  �  <  �    R  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  U  7     �   �  �  �  �  �  �  �  �  �  �  w  `  G  )  
  �  �  �  �  f  /  ?  a  k  j  \  H  /    �  �  �  �  r  b  u  v  `    �  B  u  �  �  �  �  �  �  �  �  �  w  _  A  "    �  �  �  ^  +  �  �  �  �  �  �  �  �  �  |  o  i  c  Z  N  B  5  &    	  �  �               *  4  9  7  4  !    �  �  �  �  �  �  �  �  �  �  �  �  l  C    �  �  W  �  �     �  &   �   1  �  �    J  j  o  l  a  L  .  
  �  �  �  o  6  �  r    �  �  �    3  L  d  r  q  g  Y  ?    �  �  c    �  �     �  �  �  �  �  �  �  �  �  �  �  �  w  k  ]  L  ;  +    	   �  m  s  |  �  �  �  �  z  l  \  H  1    �  �  �  �  w  b  Q  �  �  �  �  �  }  a  D  !  �  �  �  �  p  H    �  �  �  _    �  �  �  �  �  �  �  �  q  [  D  +    �  �  �  n  5   �  �  �  �  �  �  �  �  z  q  k  e  _  ^  `  b  d  _  W  O  H  V  T  N  D  :  -      �  �  �  �  s  R  6    �  �  �  �  �  �  X  �  �  �  H  �  �  �  z    i  |  @  �  �  %  �  	'  �  �  �  �  �  �  �  �  q  Y  @  &    �  �  �  �  �  �  �  $  O  �  �  �  �  �  �  �  �  �  �  `    �  ,  �  �  �   �  �  �  �  	  	�  	�  	�  	�  	�  	V  	
  �  g    �  �  �  �  r  b           
      �  �  �  �  �  �  �  �  �  �  v  N  &  ~  �  �  {  r  g  Y  I  8  !    �  �  �  w  �  P     �   �  �  �  �  q  U  6    �  �  �  �  �  �  �  6  �  l  �  �  >  �  �  �  �  �  �  �  z  Z  1  �  �  w  !  �  T  �  U  �  �  N  �  �  �  �  �  �  �  �  �  �  �  �  I  �  <  �  �     u  �  �  �  �  ~  {  w  t  p  m  e  Y  M  B  6  *         �  �  �  �  �  �  �  �  �  �  �  �  }  V  '  �  �  :  �  �  �  *  &  #          �  �  �  �  �  �  �  �  �  l  T  :     <  -    
  �  �  �  �  �  �  a  ?    �  �  �  �  T    �  >  $    �  �  �  �  �  �  �  �  q  i  {  _  :    �  �  ;  P  �  �  �  �  �  �  �  �  �  �  R  �  �  9  �  @  �    Z  �  �  �        
  �  �  �  �  y  @  �  �  G  �    ;  ?  |  .  �  �  �  �  �  �  t  a  D        �  O  �  |    �    Y  �  �  �  �  �  �  �  �  x  Q    �  �  E  �  �  J     
�  A  �  �  /  F  D  (  �  �  c    
�  
  	D  k  E  �  p  	  T  g  z  �  �  �  u  f  S  ?  -      �  �  �  �  c  K  3  �  |  t  f  Q  1    �  �  q  :    �  �  Q  :  @  6    �  g  I  *  	  �  �  �  l  @    �  �  �  �  g  E  "     �  �  (    
  �  �  �  �  �  z  M    �  �  �  U  #  �  �  �  |  �  �  �  �  �  �  f  H  (    �  �  �  �  g  C    �  �  @  V  O  H  '    �  �  �  �  �  j  T  ?  %    �  �  �    p  �  �  �  �  �  �  q  Z  <    �  �  w  ;  �  �  J  �  �  �    �  �  �  �  �  �  �    i  P  3    �  �  �  S  "  �  �  �  �  �  �  �  �  �  �  �  f  I  %  �  �  �  W    �  Z  �  b  d  S  ,     �  �  `  #  �  �  J  �  �  c    �  �  �  q  �  �  y  i  Y  I  :  *        �  �  �  }  &  �  `   �   m  /  �  �  �  o  Q  .    �  �  �  t  B    �  �  5  �  �  $